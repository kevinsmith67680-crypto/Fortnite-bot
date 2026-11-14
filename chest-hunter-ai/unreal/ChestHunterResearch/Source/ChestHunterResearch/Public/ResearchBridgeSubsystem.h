// Game-thread side of the research bridge: owns the loopback server, validates every
// request (ResearchProtocol), executes it against AResearchGameMode and encodes the reply.
//
// Command-line switches (privileged modes are OFF by default):
//   -ResearchPort=47820     loopback port (1024-65535)
//   -ResearchOracle         allow GET_ORACLE_STATE (oracle baseline only)
//   -ResearchEvaluation     allow GET_ORACLE_STATE and GET_EVALUATION (offline evaluation)
//   -ResearchBridgeOff      do not start the server
// The standard visual agent runs without the privileged switches; its client-side AgentPort
// cannot send privileged commands either way.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Tickable.h"
#include "ResearchProtocol.h"
#include "ResearchBridgeSubsystem.generated.h"

class AResearchGameMode;
class FResearchBridgeServer;
struct FResearchBridgeRequest;

UCLASS()
class CHESTHUNTERRESEARCH_API UResearchBridgeSubsystem : public UGameInstanceSubsystem, public FTickableGameObject
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	// FTickableGameObject: runs every frame, including while the world is paused.
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override { RETURN_QUICK_DECLARE_CYCLE_STAT(UResearchBridgeSubsystem, STATGROUP_Tickables); }
	virtual bool IsTickableWhenPaused() const override { return true; }
	virtual bool IsTickableInEditor() const override { return false; }
	virtual ETickableTickType GetTickableTickType() const override { return ETickableTickType::Always; }

private:
	struct FResult
	{
		TSharedPtr<FJsonObject> Header;
		TArray<uint8> Payload;
	};

	void Process(const TSharedPtr<FResearchBridgeRequest>& Request);
	void Finish(const TSharedPtr<FResearchBridgeRequest>& Request, const TSharedRef<FJsonObject>& Header,
	            const TArray<uint8>& Payload, bool bClose = false);
	void FinishError(const TSharedPtr<FResearchBridgeRequest>& Request, const FString& Command, int64 RequestId,
	                 const ResearchProtocol::FError& Error);
	void CompleteStep();
	AResearchGameMode* GetResearchGameMode() const;
	bool RequireEpisode(const TSharedPtr<FJsonObject>& Args, ResearchProtocol::FError& OutError) const;

	// Command handlers. Return false with OutError set to answer with an error.
	bool Hello(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool ResetEpisode(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool GetObservation(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, TArray<uint8>& Payload, ResearchProtocol::FError& OutError);
	bool BeginStep(const TSharedPtr<FResearchBridgeRequest>& Request, const TSharedPtr<FJsonObject>& Args, int64 Rid, ResearchProtocol::FError& OutError);
	bool EpisodeStatus(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool StopEpisode(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool Ping(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool NavQuery(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool NavRaycast(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool OracleState(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);
	bool Evaluation(const TSharedPtr<FJsonObject>& Args, int64 Rid, TSharedRef<FJsonObject>& Out, ResearchProtocol::FError& OutError);

	TUniquePtr<FResearchBridgeServer> Server;
	bool bOracleEnabled = false;
	bool bEvaluationEnabled = false;

	FString EpisodeId;
	FString Navigation = TEXT("navmesh");
	int32 EpisodeCounter = 0;
	int64 LastStepId = 0;
	TSharedPtr<FJsonObject> LastStepHeader;
	TArray<uint8> LastStepPayload;

	// The STEP currently being simulated (spans several frames).
	TSharedPtr<FResearchBridgeRequest> InFlight;
	int64 InFlightRid = 0;
	int64 InFlightStepId = 0;
	bool bInFlightObserve = false;
};
