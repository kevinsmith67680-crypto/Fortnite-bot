#include "ResearchBridgeSubsystem.h"

#include "ChestHunterResearch.h"
#include "ResearchBridgeServer.h"
#include "ResearchGameMode.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/App.h"
#include "Misc/CommandLine.h"
#include "Misc/ConfigCacheIni.h"
#include "Misc/Parse.h"

using namespace ResearchProtocol;

namespace
{
	const TCHAR* ServerName = TEXT("chest-hunter-unreal-server");
	const int32 SupportedSizes[][2] = { { 320, 180 }, { 480, 270 }, { 640, 360 }, { 800, 450 } };
	constexpr int64 MaxSafeInteger = 9007199254740992ll;   // 2^53, as in the Python validator

	FError Err(const TCHAR* Code, const FString& Message) { return FError{ FString(Code), Message }; }

	bool Choice(const TSharedPtr<FJsonObject>& Args, const FString& Key, const TArray<FString>& Options, FString& InOut, FError& OutError)
	{
		if (!Args->HasField(Key)) { return true; }
		FString Value;
		if (!Args->TryGetStringField(Key, Value) || !Options.Contains(Value))
		{
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s must be one of %s"), *Key, *FString::Join(Options, TEXT(", "))));
			return false;
		}
		InOut = Value;
		return true;
	}
}

void UResearchBridgeSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	const TCHAR* Cmd = FCommandLine::Get();
	if (FParse::Param(Cmd, TEXT("ResearchBridgeOff")))
	{
		return;
	}
	bEvaluationEnabled = FParse::Param(Cmd, TEXT("ResearchEvaluation"));
	bOracleEnabled = bEvaluationEnabled || FParse::Param(Cmd, TEXT("ResearchOracle"));

	int32 Port = 47820;
	float IdleTimeout = 120.f, ReadTimeout = 10.f, GameThreadTimeout = 60.f;
	GConfig->GetInt(TEXT("ResearchBridge"), TEXT("Port"), Port, GEngineIni);
	GConfig->GetFloat(TEXT("ResearchBridge"), TEXT("IdleTimeoutSeconds"), IdleTimeout, GEngineIni);
	GConfig->GetFloat(TEXT("ResearchBridge"), TEXT("ReadTimeoutSeconds"), ReadTimeout, GEngineIni);
	GConfig->GetFloat(TEXT("ResearchBridge"), TEXT("GameThreadTimeoutSeconds"), GameThreadTimeout, GEngineIni);
	FParse::Value(Cmd, TEXT("ResearchPort="), Port);

	// Deterministic stepping: every engine frame advances exactly 1/30 s of simulated time,
	// independent of how long rendering takes on this machine.
	FApp::SetUseFixedTimeStep(true);
	FApp::SetFixedDeltaTime(1.0 / 30.0);

	Server = MakeUnique<FResearchBridgeServer>(Port, IdleTimeout, ReadTimeout, GameThreadTimeout);
	FString Error;
	if (!Server->Start(Error))
	{
		UE_LOG(LogChestHunter, Error, TEXT("Research bridge not started: %s"), *Error);
		Server.Reset();
	}
	UE_LOG(LogChestHunter, Log, TEXT("Research bridge modes: oracle=%d evaluation=%d"), bOracleEnabled, bEvaluationEnabled);
}

void UResearchBridgeSubsystem::Deinitialize()
{
	if (Server.IsValid())
	{
		Server->Shutdown();
		Server.Reset();
	}
	Super::Deinitialize();
}

AResearchGameMode* UResearchBridgeSubsystem::GetResearchGameMode() const
{
	UWorld* World = GetGameInstance() != nullptr ? GetGameInstance()->GetWorld() : nullptr;
	return World != nullptr ? Cast<AResearchGameMode>(UGameplayStatics::GetGameMode(World)) : nullptr;
}

void UResearchBridgeSubsystem::Tick(float DeltaTime)
{
	if (!Server.IsValid())
	{
		return;
	}
	if (InFlight.IsValid())
	{
		AResearchGameMode* GM = GetResearchGameMode();
		if (GM == nullptr || !GM->IsStepping())
		{
			CompleteStep();
		}
		return;                                       // one request at a time, in order
	}
	// Bounded work per frame keeps the engine responsive.
	for (int32 I = 0; I < 8 && !InFlight.IsValid(); ++I)
	{
		TSharedPtr<FResearchBridgeRequest> Request = Server->PopRequest();
		if (!Request.IsValid())
		{
			break;
		}
		Process(Request);
	}
}

void UResearchBridgeSubsystem::Finish(const TSharedPtr<FResearchBridgeRequest>& Request, const TSharedRef<FJsonObject>& Header,
                                      const TArray<uint8>& Payload, bool bClose)
{
	FError Error;
	if (!EncodeMessage(Header, Payload, Request->Response, Error))
	{
		FString Command;
		Header->TryGetStringField(TEXT("command"), Command);
		EncodeMessage(ErrorResponse(Command, 0, Err(Errors::Internal, Error.Message)), TArray<uint8>(), Request->Response, Error);
	}
	Request->bCloseAfterResponse = bClose;
	Request->Complete();
}

void UResearchBridgeSubsystem::FinishError(const TSharedPtr<FResearchBridgeRequest>& Request, const FString& Command, int64 RequestId,
                                           const FError& Error)
{
	Finish(Request, ErrorResponse(Command, RequestId, Error), TArray<uint8>(), IsFatal(Error.Code));
}

void UResearchBridgeSubsystem::Process(const TSharedPtr<FResearchBridgeRequest>& Request)
{
	FString Command;
	int64 Rid = 0;
	TSharedPtr<FJsonObject> Args;
	FError Error;
	if (!ValidateEnvelope(Request->Header, Command, Rid, Args, Error))
	{
		double RidNumber = 0.0;
		Request->Header->TryGetNumberField(TEXT("request_id"), RidNumber);
		FinishError(Request, Command, int64(RidNumber), Error);
		return;
	}
	TSharedRef<FJsonObject> Out = OkResponse(Command, Rid);
	TArray<uint8> Payload;
	bool bOk = false;
	if (Command == TEXT("HELLO")) { bOk = Hello(Args, Rid, Out, Error); }
	else if (Command == TEXT("RESET_EPISODE")) { bOk = ResetEpisode(Args, Rid, Out, Error); }
	else if (Command == TEXT("GET_OBSERVATION")) { bOk = GetObservation(Args, Rid, Out, Payload, Error); }
	else if (Command == TEXT("STEP"))
	{
		if (BeginStep(Request, Args, Rid, Error)) { return; }   // answered later (or now, if a duplicate)
		FinishError(Request, Command, Rid, Error);
		return;
	}
	else if (Command == TEXT("GET_EPISODE_STATUS")) { bOk = EpisodeStatus(Args, Rid, Out, Error); }
	else if (Command == TEXT("STOP_EPISODE")) { bOk = StopEpisode(Args, Rid, Out, Error); }
	else if (Command == TEXT("PING")) { bOk = Ping(Args, Rid, Out, Error); }
	else if (Command == TEXT("NAV_QUERY")) { bOk = NavQuery(Args, Rid, Out, Error); }
	else if (Command == TEXT("NAV_RAYCAST")) { bOk = NavRaycast(Args, Rid, Out, Error); }
	else if (Command == TEXT("GET_ORACLE_STATE")) { bOk = OracleState(Args, Rid, Out, Error); }
	else if (Command == TEXT("GET_EVALUATION")) { bOk = Evaluation(Args, Rid, Out, Error); }
	else { Error = Err(Errors::UnknownCommand, Command); }
	if (bOk)
	{
		Finish(Request, Out, Payload);
	}
	else
	{
		FinishError(Request, Command, Rid, Error);
	}
}

bool UResearchBridgeSubsystem::RequireEpisode(const TSharedPtr<FJsonObject>& Args, FError& OutError) const
{
	FString Id;
	if (!GetString(Args, TEXT("episode_id"), true, Id, OutError))
	{
		return false;
	}
	const AResearchGameMode* GM = GetResearchGameMode();
	if (GM == nullptr || !GM->HasEpisode() || EpisodeId.IsEmpty())
	{
		OutError = Err(Errors::NoEpisode, TEXT("no episode has been started (RESET_EPISODE)"));
		return false;
	}
	if (Id != EpisodeId)
	{
		OutError = Err(Errors::StaleEpisode, FString::Printf(TEXT("episode '%s' is not the current episode '%s'"), *Id, *EpisodeId));
		return false;
	}
	return true;
}

bool UResearchBridgeSubsystem::Hello(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	FString Protocol, Client;
	int64 ClientVersion = 0;
	if (!CheckKeys(Args, { TEXT("protocol"), TEXT("version"), TEXT("client") }, OutError)
		|| !GetString(Args, TEXT("protocol"), true, Protocol, OutError)
		|| !GetInteger(Args, TEXT("version"), 0, 1000, true, 0, ClientVersion, OutError)
		|| !GetString(Args, TEXT("client"), false, Client, OutError))
	{
		return false;
	}
	if (Protocol != ProtocolName || ClientVersion != Version)
	{
		OutError = Err(Errors::VersionMismatch, FString::Printf(TEXT("client speaks %s v%lld, server %s v%d"), *Protocol, ClientVersion, ProtocolName, Version));
		return false;
	}
	Out->SetStringField(TEXT("server"), ServerName);
	Out->SetStringField(TEXT("engine"), FString::Printf(TEXT("unreal-%d.%d"), ENGINE_MAJOR_VERSION, ENGINE_MINOR_VERSION));
	Out->SetStringField(TEXT("protocol"), ProtocolName);
	Out->SetNumberField(TEXT("version"), Version);
	TArray<TSharedPtr<FJsonValue>> Maps;
	for (const FString& Map : UResearchArenaBuilder::AvailableMaps())
	{
		Maps.Add(MakeShared<FJsonValueString>(Map));
	}
	Out->SetArrayField(TEXT("maps"), Maps);
	TArray<TSharedPtr<FJsonValue>> Sizes;
	for (const auto& S : SupportedSizes)
	{
		Sizes.Add(MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>{ MakeShared<FJsonValueNumber>(S[0]), MakeShared<FJsonValueNumber>(S[1]) }));
	}
	Out->SetArrayField(TEXT("observation_sizes"), Sizes);
	TSharedRef<FJsonObject> Features = MakeShared<FJsonObject>();
	Features->SetBoolField(TEXT("navigation_service"), true);
	Features->SetBoolField(TEXT("navigation_raycast"), true);
	Features->SetBoolField(TEXT("oracle"), bOracleEnabled);
	Features->SetBoolField(TEXT("evaluation"), bEvaluationEnabled);
	Out->SetObjectField(TEXT("features"), Features);
	return true;
}

bool UResearchBridgeSubsystem::ResetEpisode(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	AResearchGameMode* GM = GetResearchGameMode();
	if (GM == nullptr)
	{
		OutError = Err(Errors::Internal, TEXT("the research game mode is not active (open the ResearchArena3D map)"));
		return false;
	}
	FResearchEpisodeConfig Config;
	int64 Seed = 0, Episode = 0, MaxTicks = Config.MaxTicks;
	double Fov = Config.FovDeg;
	if (!CheckKeys(Args, { TEXT("map_id"), TEXT("seed"), TEXT("episode"), TEXT("observation"), TEXT("fov_deg"),
	                       TEXT("graphics_quality"), TEXT("max_ticks"), TEXT("navigation") }, OutError)
		|| !GetString(Args, TEXT("map_id"), true, Config.MapId, OutError)
		|| !GetInteger(Args, TEXT("seed"), 0, 2147483647ll, true, 0, Seed, OutError)
		|| !GetInteger(Args, TEXT("episode"), 0, 2147483647ll, false, 0, Episode, OutError)
		|| !GetNumber(Args, TEXT("fov_deg"), 60.0, 110.0, false, Config.FovDeg, Fov, OutError)
		|| !GetInteger(Args, TEXT("max_ticks"), 1, 10000000, false, Config.MaxTicks, MaxTicks, OutError)
		|| !Choice(Args, TEXT("graphics_quality"), { TEXT("low"), TEXT("medium"), TEXT("high") }, Config.GraphicsQuality, OutError)
		|| !Choice(Args, TEXT("navigation"), { TEXT("navmesh"), TEXT("local"), TEXT("none") }, Config.Navigation, OutError))
	{
		return false;
	}
	const TSharedPtr<FJsonObject>* Size = nullptr;
	if (Args->HasField(TEXT("observation")))
	{
		int64 W = 0, H = 0;
		if (!Args->TryGetObjectField(TEXT("observation"), Size)
			|| !GetInteger(*Size, TEXT("width"), 1, 10000, true, 0, W, OutError)
			|| !GetInteger(*Size, TEXT("height"), 1, 10000, true, 0, H, OutError))
		{
			OutError = Err(Errors::InvalidArgument, TEXT("observation must be {width, height}"));
			return false;
		}
		bool bSupported = false;
		for (const auto& S : SupportedSizes) { bSupported |= (S[0] == W && S[1] == H); }
		if (!bSupported)
		{
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("observation size %lldx%lld not supported (320x180, 480x270, 640x360, 800x450)"), W, H));
			return false;
		}
		Config.Width = int32(W);
		Config.Height = int32(H);
	}
	Config.Seed = int32(Seed);
	Config.Episode = int32(Episode);
	Config.FovDeg = float(Fov);
	Config.MaxTicks = MaxTicks;
	GM->SetPrivilegeModes(bOracleEnabled, bEvaluationEnabled);
	FString Error;
	if (!GM->ResetEpisode(Config, Error))
	{
		OutError = Err(Errors::InvalidArgument, Error);
		return false;
	}
	++EpisodeCounter;
	EpisodeId = FString::Printf(TEXT("%d-%d-%d"), Config.Seed, Config.Episode, EpisodeCounter);
	Navigation = Config.Navigation;
	LastStepId = 0;
	LastStepHeader.Reset();
	LastStepPayload.Reset();
	Out->SetStringField(TEXT("episode_id"), EpisodeId);
	Out->SetObjectField(TEXT("rules"), GM->RulesJson());
	Out->SetStringField(TEXT("navigation"), Navigation);
	Out->SetObjectField(TEXT("telemetry"), GM->Telemetry());
	return true;
}

bool UResearchBridgeSubsystem::GetObservation(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out,
                                              TArray<uint8>& Payload, FError& OutError)
{
	if (!CheckKeys(Args, { TEXT("episode_id") }, OutError) || !RequireEpisode(Args, OutError))
	{
		return false;
	}
	TSharedPtr<FJsonObject> Meta;
	if (!GetResearchGameMode()->Observe(Meta, Payload))
	{
		OutError = Err(Errors::Internal, TEXT("camera capture failed"));
		return false;
	}
	Out->SetStringField(TEXT("episode_id"), EpisodeId);
	Out->SetNumberField(TEXT("step_id"), double(LastStepId));
	Out->SetObjectField(TEXT("observation"), Meta);
	return true;
}

bool UResearchBridgeSubsystem::BeginStep(const TSharedPtr<FResearchBridgeRequest>& Request, const TSharedPtr<FJsonObject>& Args,
                                         int64 Rid, FError& OutError)
{
	int64 StepId = 0, Ticks = 1;
	bool bObserve = false;
	FAction Action;
	const TSharedPtr<FJsonObject>* ActionObject = nullptr;
	if (!CheckKeys(Args, { TEXT("episode_id"), TEXT("step_id"), TEXT("action"), TEXT("ticks"), TEXT("observe") }, OutError)
		|| !GetInteger(Args, TEXT("step_id"), 1, MaxSafeInteger, true, 0, StepId, OutError)
		|| !GetInteger(Args, TEXT("ticks"), 1, MaxTicksPerStep, false, 1, Ticks, OutError)
		|| !GetBool(Args, TEXT("observe"), false, bObserve, OutError))
	{
		return false;
	}
	if (!Args->TryGetObjectField(TEXT("action"), ActionObject))
	{
		OutError = Err(Errors::InvalidAction, TEXT("action must be a JSON object"));
		return false;
	}
	if (!ValidateAction(*ActionObject, Action, OutError) || !RequireEpisode(Args, OutError))
	{
		return false;
	}
	if (StepId == LastStepId && LastStepHeader.IsValid())
	{
		// A resend after a reconnect: answer from the cache, never apply the action twice.
		TSharedRef<FJsonObject> Copy = MakeShared<FJsonObject>(*LastStepHeader);
		Copy->SetNumberField(TEXT("request_id"), double(Rid));
		Copy->SetBoolField(TEXT("duplicate"), true);
		Finish(Request, Copy, LastStepPayload);
		return true;
	}
	if (StepId <= LastStepId)
	{
		OutError = Err(Errors::StaleStep, FString::Printf(TEXT("step %lld is older than %lld"), StepId, LastStepId));
		return false;
	}
	if (StepId != LastStepId + 1)
	{
		OutError = Err(Errors::OutOfOrder, FString::Printf(TEXT("expected step %lld, got %lld"), LastStepId + 1, StepId));
		return false;
	}
	AResearchGameMode* GM = GetResearchGameMode();
	if (GM->IsEnded())
	{
		OutError = Err(Errors::EpisodeEnded, TEXT("the episode has ended"));
		return false;
	}
	InFlight = Request;
	InFlightRid = Rid;
	InFlightStepId = StepId;
	bInFlightObserve = bObserve;
	GM->BeginStep(Action, int32(Ticks));               // unpauses; the world advances Ticks fixed frames
	return true;
}

void UResearchBridgeSubsystem::CompleteStep()
{
	TSharedPtr<FResearchBridgeRequest> Request = MoveTemp(InFlight);
	InFlight.Reset();
	AResearchGameMode* GM = GetResearchGameMode();
	if (GM == nullptr)
	{
		FinishError(Request, TEXT("STEP"), InFlightRid, Err(Errors::Internal, TEXT("the game mode disappeared during the step")));
		return;
	}
	LastStepId = InFlightStepId;
	TSharedRef<FJsonObject> Out = OkResponse(TEXT("STEP"), InFlightRid);
	Out->SetStringField(TEXT("episode_id"), EpisodeId);
	Out->SetNumberField(TEXT("step_id"), double(InFlightStepId));
	Out->SetObjectField(TEXT("telemetry"), GM->Telemetry());
	Out->SetBoolField(TEXT("ended"), GM->IsEnded());
	Out->SetBoolField(TEXT("duplicate"), false);
	TArray<uint8> Payload;
	if (bInFlightObserve)
	{
		TSharedPtr<FJsonObject> Meta;
		if (GM->Observe(Meta, Payload))
		{
			Out->SetObjectField(TEXT("observation"), Meta);
		}
		else
		{
			Payload.Reset();
		}
	}
	LastStepHeader = MakeShared<FJsonObject>(*Out);
	LastStepPayload = Payload;
	Finish(Request, Out, Payload);
}

bool UResearchBridgeSubsystem::EpisodeStatus(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	if (!CheckKeys(Args, { TEXT("episode_id") }, OutError) || !RequireEpisode(Args, OutError))
	{
		return false;
	}
	const AResearchGameMode* GM = GetResearchGameMode();
	Out->SetStringField(TEXT("episode_id"), EpisodeId);
	Out->SetStringField(TEXT("state"), GM->IsStopped() ? TEXT("stopped") : (GM->IsEnded() ? TEXT("ended") : TEXT("running")));
	Out->SetNumberField(TEXT("tick"), double(GM->GetTick()));
	Out->SetNumberField(TEXT("sim_time"), GM->GetSimTime());
	Out->SetNumberField(TEXT("max_ticks"), double(GM->GetMaxTicks()));
	Out->SetNumberField(TEXT("last_step_id"), double(LastStepId));
	return true;
}

bool UResearchBridgeSubsystem::StopEpisode(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	FString Reason;
	if (!CheckKeys(Args, { TEXT("episode_id"), TEXT("reason") }, OutError) || !GetString(Args, TEXT("reason"), false, Reason, OutError)
		|| !RequireEpisode(Args, OutError))
	{
		return false;
	}
	AResearchGameMode* GM = GetResearchGameMode();
	GM->Stop();
	Out->SetStringField(TEXT("episode_id"), EpisodeId);
	Out->SetStringField(TEXT("state"), TEXT("stopped"));
	Out->SetObjectField(TEXT("summary"), GM->Summary());
	return true;
}

bool UResearchBridgeSubsystem::Ping(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	int64 Nonce = 0;
	if (!CheckKeys(Args, { TEXT("nonce") }, OutError) || !GetInteger(Args, TEXT("nonce"), 0, MaxSafeInteger, false, 0, Nonce, OutError))
	{
		return false;
	}
	Out->SetBoolField(TEXT("pong"), true);
	if (Args->HasField(TEXT("nonce"))) { Out->SetNumberField(TEXT("nonce"), double(Nonce)); }
	else { Out->SetField(TEXT("nonce"), MakeShared<FJsonValueNull>()); }
	Out->SetNumberField(TEXT("server_time"), FDateTime::UtcNow().ToUnixTimestamp());
	return true;
}

bool UResearchBridgeSubsystem::NavQuery(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	FVector Start, Goal;
	double Tolerance = 0.5;
	if (!CheckKeys(Args, { TEXT("episode_id"), TEXT("start"), TEXT("goal"), TEXT("goal_tolerance") }, OutError)
		|| !GetVector(Args, TEXT("start"), Start, OutError) || !GetVector(Args, TEXT("goal"), Goal, OutError)
		|| !GetNumber(Args, TEXT("goal_tolerance"), 0.1, 5.0, false, 0.5, Tolerance, OutError)
		|| !RequireEpisode(Args, OutError))
	{
		return false;
	}
	if (Navigation != TEXT("navmesh"))
	{
		OutError = Err(Errors::NavigationDisabled, TEXT("the navigation service is disabled for this episode"));
		return false;
	}
	// The query takes the agent's ESTIMATED start and goal; the answer is a route, nothing about chests.
	const TSharedRef<FJsonObject> Path = GetResearchGameMode()->FindPath(Start, Goal, float(Tolerance));
	for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Path->Values)
	{
		Out->SetField(Pair.Key, Pair.Value);
	}
	return true;
}

bool UResearchBridgeSubsystem::NavRaycast(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	FVector Origin;
	double Reach = 8.0;
	const TArray<TSharedPtr<FJsonValue>>* Bearings = nullptr;
	if (!CheckKeys(Args, { TEXT("episode_id"), TEXT("origin"), TEXT("bearings_deg"), TEXT("max_distance") }, OutError)
		|| !GetVector(Args, TEXT("origin"), Origin, OutError)
		|| !GetNumber(Args, TEXT("max_distance"), 0.5, 15.0, false, 8.0, Reach, OutError))
	{
		return false;
	}
	if (!Args->TryGetArrayField(TEXT("bearings_deg"), Bearings) || Bearings->Num() < 1 || Bearings->Num() > MaxRays)
	{
		OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("bearings_deg must be a list of 1-%d angles"), MaxRays));
		return false;
	}
	for (const TSharedPtr<FJsonValue>& B : *Bearings)
	{
		if (B->Type != EJson::Number || !FMath::IsFinite(B->AsNumber()) || FMath::Abs(B->AsNumber()) > 720.0)
		{
			OutError = Err(Errors::InvalidArgument, TEXT("bearings_deg entries must be numbers within [-720, 720]"));
			return false;
		}
	}
	if (!RequireEpisode(Args, OutError))
	{
		return false;
	}
	if (Navigation == TEXT("none"))
	{
		OutError = Err(Errors::NavigationDisabled, TEXT("the navigation service is disabled for this episode"));
		return false;
	}
	TArray<TSharedPtr<FJsonValue>> Distances;
	for (const TSharedPtr<FJsonValue>& B : *Bearings)
	{
		Distances.Add(MakeShared<FJsonValueNumber>(GetResearchGameMode()->NavRaycast(Origin, float(B->AsNumber()), float(Reach))));
	}
	Out->SetArrayField(TEXT("distances"), Distances);
	Out->SetNumberField(TEXT("max_distance"), Reach);
	return true;
}

bool UResearchBridgeSubsystem::OracleState(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	if (!bOracleEnabled)
	{
		OutError = Err(Errors::Forbidden, TEXT("the server was not started in oracle or evaluation mode"));
		return false;
	}
	if (!CheckKeys(Args, { TEXT("episode_id") }, OutError) || !RequireEpisode(Args, OutError))
	{
		return false;
	}
	Out->SetObjectField(TEXT("state"), GetResearchGameMode()->OracleState());
	return true;
}

bool UResearchBridgeSubsystem::Evaluation(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, FError& OutError)
{
	if (!bEvaluationEnabled)
	{
		OutError = Err(Errors::Forbidden, TEXT("the server was not started in evaluation mode"));
		return false;
	}
	int64 FrameId = 0;
	if (!CheckKeys(Args, { TEXT("episode_id"), TEXT("frame_id") }, OutError)
		|| !GetInteger(Args, TEXT("frame_id"), 1, MaxSafeInteger, true, 0, FrameId, OutError)
		|| !RequireEpisode(Args, OutError))
	{
		return false;
	}
	const TSharedPtr<FJsonObject> Labels = GetResearchGameMode()->EvaluationLabels();
	double LabelFrame = -1.0;
	if (!Labels.IsValid() || !Labels->TryGetNumberField(TEXT("frame_id"), LabelFrame) || int64(LabelFrame) != FrameId)
	{
		OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("no labels for frame %lld (only the latest rendered frame is labelled)"), FrameId));
		return false;
	}
	Out->SetObjectField(TEXT("labels"), Labels);
	return true;
}
