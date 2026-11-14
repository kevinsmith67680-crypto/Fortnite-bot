// Loopback-only TCP server for the research bridge (runs on its own thread).
//
// * binds 127.0.0.1 only and drops any peer whose address is not loopback;
// * one client at a time (others get a `busy` error and are disconnected);
// * reads framed messages with size limits and timeouts, then hands each parsed request to
//   the game thread (UResearchBridgeSubsystem) and waits for its encoded response;
// * offers a fixed command set only: no console, no script execution, no file access.

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "Containers/Queue.h"
#include "Dom/JsonObject.h"
#include <atomic>

class FSocket;
class FRunnableThread;

/** One request travelling from the socket thread to the game thread and back. */
struct FResearchBridgeRequest
{
	TSharedPtr<FJsonObject> Header;
	TArray<uint8> Response;              // fully encoded response message (set by the game thread)
	bool bCloseAfterResponse = false;
	FEvent* Done = nullptr;
	/** Hand-over flag: whichever side sets it second owns returning Done to the event pool
	 *  (the game thread when it finished, or the socket thread when it gave up waiting). */
	std::atomic<bool> bClaimed{ false };

	/** Game thread: publish the response (or release the event if the client gave up). */
	void Complete()
	{
		if (bClaimed.exchange(true))
		{
			FPlatformProcess::ReturnSynchEventToPool(Done);   // the socket thread timed out
			Done = nullptr;
		}
		else
		{
			Done->Trigger();
		}
	}
};

class CHESTHUNTERRESEARCH_API FResearchBridgeServer : public FRunnable
{
public:
	FResearchBridgeServer(int32 InPort, float InIdleTimeoutSeconds, float InReadTimeoutSeconds, float InGameThreadTimeoutSeconds);
	virtual ~FResearchBridgeServer() override;

	bool Start(FString& OutError);
	void Shutdown();

	/** Game thread: next request to process (nullptr when none). */
	TSharedPtr<FResearchBridgeRequest> PopRequest();
	int32 GetPort() const { return Port; }

	// FRunnable
	virtual uint32 Run() override;
	virtual void Stop() override { bStopping = true; }

private:
	void ServeClient(FSocket* Client);
	bool RecvExact(FSocket* Client, uint8* Out, int32 Count, float TimeoutSeconds);
	bool SendAll(FSocket* Client, const TArray<uint8>& Bytes);
	void RejectPendingConnections();
	bool IsLoopbackPeer(FSocket* Client) const;
	void SendError(FSocket* Client, const FString& Code, const FString& Message, const FString& Command = FString(), int64 RequestId = 0);

	int32 Port;
	float IdleTimeoutSeconds;
	float ReadTimeoutSeconds;
	float GameThreadTimeoutSeconds;
	FSocket* Listener = nullptr;
	FRunnableThread* Thread = nullptr;
	std::atomic<bool> bStopping{ false };
	TQueue<TSharedPtr<FResearchBridgeRequest>, EQueueMode::Spsc> Inbox;
};
