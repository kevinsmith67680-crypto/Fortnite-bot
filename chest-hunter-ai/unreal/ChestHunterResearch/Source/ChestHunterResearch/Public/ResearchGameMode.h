// Episode logic of the research arena (game thread only). Mirrors
// chest_hunter.simulation3d.environment3d.Environment3D so that the Python agent sees the
// same contract from the reference arena and from Unreal:
//  * chest presence: UnitRoll(seed, episode, spawn index) < spawn probability;
//  * interaction: hold `interact` for interaction_seconds within radius/cone/line of sight;
//  * agent outputs: camera frame, camera rig, drifting odometry, notification events;
//  * privileged outputs (oracle/evaluation command-line modes only): truth and labels.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "ResearchArenaBuilder.h"
#include "ResearchProtocol.h"
#include "ResearchGameMode.generated.h"

class AResearchCharacter;
class AResearchChest;

struct FResearchEpisodeConfig
{
	FString MapId = TEXT("research_arena_3d");
	int32 Seed = 0;
	int32 Episode = 0;
	int32 Width = 640;
	int32 Height = 360;
	float FovDeg = 90.f;
	FString GraphicsQuality = TEXT("low");
	int64 MaxTicks = 18000;
	FString Navigation = TEXT("navmesh");
};

struct FResearchRules
{
	int32 TickRate = 30;
	float InteractionRadius = 1.8f;
	float InteractionConeDeg = 60.f;
	float InteractionSeconds = 2.0f;
	float InteractionFailureProbability = 0.f;
	float RewardProbability = 0.3f;
	float OdometryDrift = 0.005f;
};

UCLASS()
class CHESTHUNTERRESEARCH_API AResearchGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	AResearchGameMode();

	virtual void Tick(float DeltaSeconds) override;

	// ---- episode control (called by UResearchBridgeSubsystem on the game thread) ----
	bool ResetEpisode(const FResearchEpisodeConfig& Config, FString& OutError);
	void BeginStep(const ResearchProtocol::FAction& Action, int32 Ticks);
	bool IsStepping() const { return StepTicksRemaining > 0; }
	bool HasEpisode() const { return Character != nullptr && bHasEpisode; }
	bool IsEnded() const { return bStopped || TickCount >= Config.MaxTicks; }
	void Stop() { bStopped = true; }
	bool IsStopped() const { return bStopped; }
	int64 GetTick() const { return TickCount; }
	double GetSimTime() const { return double(TickCount) / Rules.TickRate; }
	int64 GetMaxTicks() const { return Config.MaxTicks; }
	const FResearchRules& GetRules() const { return Rules; }
	void SetPrivilegeModes(bool bOracle, bool bEvaluation) { bOracleEnabled = bOracle; bEvaluationEnabled = bEvaluation; }

	// ---- agent-facing outputs ----
	TSharedRef<FJsonObject> RulesJson() const;
	TSharedRef<FJsonObject> Telemetry();
	bool Observe(TSharedPtr<FJsonObject>& OutMeta, TArray<uint8>& OutPixels);
	TSharedRef<FJsonObject> Summary() const;

	// ---- navigation service (estimated positions in, path out; never chest positions) ----
	TSharedRef<FJsonObject> FindPath(const FVector& StartM, const FVector& GoalM, float ToleranceM) const;
	float NavRaycast(const FVector& OriginM, float BearingDeg, float MaxDistanceM) const;

	// ---- privileged (refused by the bridge unless the matching mode is enabled) ----
	TSharedRef<FJsonObject> OracleState() const;
	TSharedPtr<FJsonObject> EvaluationLabels() const { return LastLabels; }

protected:
	virtual void BeginPlay() override;

private:
	AResearchChest* InteractableChest() const;
	void UpdateInteraction(bool bHeld);
	void UpdateCues();
	TSharedRef<FJsonObject> IntrinsicsJson() const;
	TSharedPtr<FJsonObject> ComputeLabels() const;
	void EnsureLighting();

	FResearchEpisodeConfig Config;
	FResearchRules Rules;
	FResearchArenaLayout Layout;
	TSharedPtr<FJsonObject> LayoutJson;
	FString BuiltMapId;

	UPROPERTY(Transient) TArray<TObjectPtr<AActor>> ArenaActors;
	UPROPERTY(Transient) TArray<TObjectPtr<AResearchChest>> Chests;
	UPROPERTY(Transient) TObjectPtr<AResearchCharacter> Character;

	bool bHasEpisode = false;
	bool bStopped = false;
	bool bOracleEnabled = false;
	bool bEvaluationEnabled = false;
	int64 TickCount = 0;
	int32 StepTicksRemaining = 0;
	bool bStepInteract = false;
	int32 HoldTarget = INDEX_NONE;
	int32 HoldTicks = 0;
	int32 FailedAttempts = 0;
	int32 RewardCount = 0;
	int64 FrameId = 0;
	TMap<int32, int64> RewardFlashUntil;
	TArray<TSharedPtr<FJsonValue>> PendingEvents;
	TArray<TSharedPtr<FJsonValue>> OpenedLog;
	TSharedPtr<FJsonObject> LastLabels;
};
