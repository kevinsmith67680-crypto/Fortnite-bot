#include "ResearchBridgeServer.h"

#include "ChestHunterResearch.h"
#include "ResearchProtocol.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/RunnableThread.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "IPAddress.h"
#include "Sockets.h"
#include "SocketSubsystem.h"

FResearchBridgeServer::FResearchBridgeServer(int32 InPort, float InIdle, float InRead, float InGameThread)
	: Port(InPort), IdleTimeoutSeconds(InIdle), ReadTimeoutSeconds(InRead), GameThreadTimeoutSeconds(InGameThread)
{
}

FResearchBridgeServer::~FResearchBridgeServer()
{
	Shutdown();
}

bool FResearchBridgeServer::Start(FString& OutError)
{
	if (Port < 1024 || Port > 65535)
	{
		OutError = FString::Printf(TEXT("port %d outside 1024-65535"), Port);
		return false;
	}
	// Loopback ONLY: the research service must never be reachable from other machines.
	const FIPv4Endpoint Endpoint(FIPv4Address(127, 0, 0, 1), uint16(Port));
	Listener = FTcpSocketBuilder(TEXT("ChestHunterResearchBridge"))
		.AsReusable()
		.AsBlocking()
		.BoundToEndpoint(Endpoint)
		.Listening(2)
		.Build();
	if (Listener == nullptr)
	{
		OutError = FString::Printf(TEXT("could not listen on 127.0.0.1:%d (port in use?)"), Port);
		return false;
	}
	Thread = FRunnableThread::Create(this, TEXT("ChestHunterResearchBridge"), 0, TPri_Normal);
	UE_LOG(LogChestHunter, Log, TEXT("Research bridge listening on 127.0.0.1:%d"), Port);
	return Thread != nullptr;
}

void FResearchBridgeServer::Shutdown()
{
	bStopping = true;
	if (Thread != nullptr)
	{
		Thread->WaitForCompletion();
		delete Thread;
		Thread = nullptr;
	}
	if (Listener != nullptr)
	{
		Listener->Close();
		ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Listener);
		Listener = nullptr;
	}
}

TSharedPtr<FResearchBridgeRequest> FResearchBridgeServer::PopRequest()
{
	TSharedPtr<FResearchBridgeRequest> Request;
	return Inbox.Dequeue(Request) ? Request : nullptr;
}

bool FResearchBridgeServer::IsLoopbackPeer(FSocket* Client) const
{
	TSharedRef<FInternetAddr> Peer = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
	Client->GetPeerAddress(*Peer);
	uint32 Ip = 0;
	Peer->GetIp(Ip);
	return (Ip >> 24) == 127;                           // 127.0.0.0/8
}

uint32 FResearchBridgeServer::Run()
{
	while (!bStopping)
	{
		bool bPending = false;
		if (!Listener->WaitForPendingConnection(bPending, FTimespan::FromMilliseconds(200)) || !bPending)
		{
			continue;
		}
		FSocket* Client = Listener->Accept(TEXT("ChestHunterResearchClient"));
		if (Client == nullptr)
		{
			continue;
		}
		if (!IsLoopbackPeer(Client))
		{
			UE_LOG(LogChestHunter, Warning, TEXT("Rejected a non-loopback connection"));
			Client->Close();
			ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Client);
			continue;
		}
		Client->SetNoDelay(true);
		ServeClient(Client);
		Client->Close();
		ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Client);
	}
	return 0;
}

void FResearchBridgeServer::RejectPendingConnections()
{
	bool bPending = false;
	while (Listener->HasPendingConnection(bPending) && bPending)
	{
		FSocket* Other = Listener->Accept(TEXT("ChestHunterResearchRejected"));
		if (Other == nullptr)
		{
			break;
		}
		SendError(Other, ResearchProtocol::Errors::Busy, TEXT("another client is connected"));
		Other->Close();
		ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Other);
	}
}

bool FResearchBridgeServer::RecvExact(FSocket* Client, uint8* Out, int32 Count, float TimeoutSeconds)
{
	int32 Got = 0;
	double Deadline = FPlatformTime::Seconds() + TimeoutSeconds;
	while (Got < Count)
	{
		if (bStopping)
		{
			return false;
		}
		if (!Client->Wait(ESocketWaitConditions::WaitForRead, FTimespan::FromMilliseconds(200)))
		{
			if (Got == 0) { RejectPendingConnections(); }
			if (FPlatformTime::Seconds() > Deadline)
			{
				return false;
			}
			continue;
		}
		int32 Read = 0;
		if (!Client->Recv(Out + Got, Count - Got, Read) || Read <= 0)
		{
			return false;                               // closed by the client
		}
		Got += Read;
		Deadline = FPlatformTime::Seconds() + ReadTimeoutSeconds;   // the rest must follow promptly
	}
	return true;
}

bool FResearchBridgeServer::SendAll(FSocket* Client, const TArray<uint8>& Bytes)
{
	int32 Sent = 0;
	while (Sent < Bytes.Num())
	{
		int32 N = 0;
		if (!Client->Send(Bytes.GetData() + Sent, Bytes.Num() - Sent, N) || N <= 0)
		{
			return false;
		}
		Sent += N;
	}
	return true;
}

void FResearchBridgeServer::SendError(FSocket* Client, const FString& Code, const FString& Message, const FString& Command, int64 RequestId)
{
	ResearchProtocol::FError Error{ Code, Message };
	TArray<uint8> Bytes;
	ResearchProtocol::FError EncodeError;
	if (ResearchProtocol::EncodeMessage(ResearchProtocol::ErrorResponse(Command, RequestId, Error), TArray<uint8>(), Bytes, EncodeError))
	{
		SendAll(Client, Bytes);
	}
}

void FResearchBridgeServer::ServeClient(FSocket* Client)
{
	using namespace ResearchProtocol;
	bool bGreeted = false;
	TArray<uint8> Buffer;
	while (!bStopping)
	{
		uint8 PrefixBytes[PrefixSize];
		if (!RecvExact(Client, PrefixBytes, PrefixSize, IdleTimeoutSeconds))
		{
			return;
		}
		FPrefix Prefix;
		FError Error;
		if (!ParsePrefix(PrefixBytes, PrefixSize, Prefix, Error))
		{
			SendError(Client, Error.Code, Error.Message);
			return;                                       // the stream cannot be trusted any more
		}
		Buffer.SetNumUninitialized(Prefix.HeaderLength);
		if (!RecvExact(Client, Buffer.GetData(), Prefix.HeaderLength, ReadTimeoutSeconds))
		{
			return;
		}
		TSharedPtr<FJsonObject> Header;
		if (!ParseHeader(Buffer.GetData(), Prefix.HeaderLength, Header, Error))
		{
			SendError(Client, Error.Code, Error.Message);
			return;
		}
		if (Prefix.PayloadLength > 0)
		{
			// Requests never carry payloads; read and discard to keep the stream aligned.
			Buffer.SetNumUninitialized(Prefix.PayloadLength);
			if (!RecvExact(Client, Buffer.GetData(), Prefix.PayloadLength, ReadTimeoutSeconds))
			{
				return;
			}
		}
		FString Command;
		Header->TryGetStringField(TEXT("command"), Command);
		double RidNumber = 0.0;
		const int64 Rid = Header->TryGetNumberField(TEXT("request_id"), RidNumber) ? int64(RidNumber) : 0;
		if (!bGreeted && Command != TEXT("HELLO") && Command != TEXT("PING"))
		{
			SendError(Client, Errors::Forbidden, TEXT("send HELLO first (protocol version negotiation)"), Command, Rid);
			continue;
		}
		TSharedPtr<FResearchBridgeRequest> Request = MakeShared<FResearchBridgeRequest>();
		Request->Header = Header;
		Request->Done = FPlatformProcess::GetSynchEventFromPool(true);
		Inbox.Enqueue(Request);
		bool bAnswered = Request->Done->Wait(FTimespan::FromSeconds(GameThreadTimeoutSeconds));
		if (!bAnswered && Request->bClaimed.exchange(true))
		{
			bAnswered = true;                             // completed just as the wait timed out
		}
		if (!bAnswered)
		{
			// The game thread still owns the request and will release the event when it is done.
			SendError(Client, TEXT("timeout"), TEXT("the game thread did not answer in time"), Command, Rid);
			continue;
		}
		FPlatformProcess::ReturnSynchEventToPool(Request->Done);
		Request->Done = nullptr;
		if (Command == TEXT("HELLO") && !Request->bCloseAfterResponse)
		{
			// The game thread marks a refused HELLO as close-after-response; otherwise greeted.
			TSharedPtr<FJsonObject> Parsed;
			FPrefix RespPrefix;
			if (ParsePrefix(Request->Response.GetData(), PrefixSize, RespPrefix, Error)
				&& ParseHeader(Request->Response.GetData() + PrefixSize, RespPrefix.HeaderLength, Parsed, Error))
			{
				bGreeted = Parsed->GetStringField(TEXT("status")) == TEXT("ok");
			}
		}
		if (!SendAll(Client, Request->Response) || Request->bCloseAfterResponse)
		{
			return;
		}
	}
}
