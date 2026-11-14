#include "ResearchGameMode.h"

#include "ChestHunterResearch.h"
#include "ResearchCharacter.h"
#include "ResearchChest.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Engine/DirectionalLight.h"
#include "Engine/SkyLight.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "Kismet/GameplayStatics.h"
#include "NavigationPath.h"
#include "NavigationSystem.h"

namespace
{
	double BearingDeg(double Dx, double Dy)
	{
		return FMath::Fmod(FMath::RadiansToDegrees(FMath::Atan2(Dy, Dx)) + 360.0, 360.0);
	}
	double AngleDifferenceDeg(double A, double B)
	{
		return FMath::Fmod(A - B + 540.0, 360.0) - 180.0;
	}
	TArray<TSharedPtr<FJsonValue>> Vec(const FVector& V)
	{
		return { MakeShared<FJsonValueNumber>(V.X), MakeShared<FJsonValueNumber>(V.Y), MakeShared<FJsonValueNumber>(V.Z) };
	}
}

AResearchGameMode::AResearchGameMode()
{
	DefaultPawnClass = AResearchCharacter::StaticClass();
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PostPhysics;      // after the character moved this tick
	PrimaryActorTick.bTickEvenWhenPaused = false;
}

void AResearchGameMode::BeginPlay()
{
	Super::BeginPlay();
	EnsureLighting();
	// Between bridge steps the world is paused: simulated time only advances on STEP.
	UGameplayStatics::SetGamePaused(this, true);
}

void AResearchGameMode::EnsureLighting()
{
	UWorld* World = GetWorld();
	if (TActorIterator<ADirectionalLight>(World))
	{
		return;                                       // the map provides its own lighting
	}
	ADirectionalLight* Sun = World->SpawnActor<ADirectionalLight>(FVector(0, 0, 5000), FRotator(-50.f, 30.f, 0.f));
	if (Sun != nullptr)
	{
		Sun->GetLightComponent()->SetMobility(EComponentMobility::Movable);
		Sun->GetLightComponent()->SetIntensity(6.f);
	}
	ASkyLight* Sky = World->SpawnActor<ASkyLight>(FVector(0, 0, 3000), FRotator::ZeroRotator);
	if (Sky != nullptr)
	{
		Sky->GetLightComponent()->SetMobility(EComponentMobility::Movable);
		Sky->GetLightComponent()->bRealTimeCapture = true;
		Sky->GetLightComponent()->SetIntensity(1.f);
	}
}

bool AResearchGameMode::ResetEpisode(const FResearchEpisodeConfig& InConfig, FString& OutError)
{
	UWorld* World = GetWorld();
	if (BuiltMapId != InConfig.MapId)
	{
		FResearchArenaLayout NewLayout;
		TSharedPtr<FJsonObject> NewJson;
		if (!UResearchArenaBuilder::LoadLayout(InConfig.MapId, NewLayout, NewJson, OutError))
		{
			return false;
		}
		for (AActor* A : ArenaActors) { if (A) { A->Destroy(); } }
		ArenaActors.Reset();
		for (AResearchChest* C : Chests) { if (C) { C->Destroy(); } }
		Chests.Reset();
		TArray<AActor*> Spawned;
		if (!UResearchArenaBuilder::BuildStatic(World, NewJson, Spawned, OutError))
		{
			return false;
		}
		ArenaActors.Append(Spawned);
		for (const FResearchSpawnPoint& S : NewLayout.Spawns)
		{
			FActorSpawnParameters Params;
			Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
			AResearchChest* Chest = World->SpawnActor<AResearchChest>(ResearchUnits::ToCm(S.PositionM),
			                                                           FRotator(0.f, S.YawDeg, 0.f), Params);
			Chests.Add(Chest);
		}
		Layout = NewLayout;
		LayoutJson = NewJson;
		BuiltMapId = InConfig.MapId;
	}
	Config = InConfig;
	TickCount = 0;
	StepTicksRemaining = 0;
	HoldTarget = INDEX_NONE;
	HoldTicks = 0;
	FailedAttempts = 0;
	RewardCount = 0;
	FrameId = 0;
	bStopped = false;
	RewardFlashUntil.Reset();
	PendingEvents.Reset();
	OpenedLog.Reset();
	LastLabels.Reset();

	// Seeded presence: identical to the Python reference (environment3d.unit_roll).
	for (int32 I = 0; I < Chests.Num(); ++I)
	{
		const FResearchSpawnPoint& S = Layout.Spawns[I];
		const bool bPresent = ResearchProtocol::UnitRoll(Config.Seed, Config.Episode, I, 0) < S.Probability;
		Chests[I]->SetActorLocation(ResearchUnits::ToCm(S.PositionM));
		Chests[I]->Initialise(I, S.LocationName, S.YawDeg, bPresent, Layout.ChestValue, Layout.ChestHueShiftDeg);
	}

	// Procedural lighting variation: scale the sun (base intensity 6, see EnsureLighting).
	for (TActorIterator<ADirectionalLight> It(World); It; ++It)
	{
		It->GetLightComponent()->SetIntensity(6.f * Layout.Lighting);
	}

	APlayerController* PC = UGameplayStatics::GetPlayerController(this, 0);
	Character = PC != nullptr ? Cast<AResearchCharacter>(PC->GetPawn()) : nullptr;
	if (Character == nullptr)
	{
		OutError = TEXT("the research character has not been spawned (check GlobalDefaultGameMode)");
		return false;
	}
	PC->SetIgnoreMoveInput(true);                     // no local keyboard/mouse control of the agent
	PC->SetIgnoreLookInput(true);
	Character->ResetTo(ResearchUnits::ToCm(Layout.StartM), Layout.StartYawDeg);
	Character->ConfigureCamera(Config.Width, Config.Height, Config.FovDeg);
	Character->ResetOdometry(ResearchProtocol::SplitMix64(uint64(Config.Seed) * 1000003ull + uint64(Config.Episode)),
	                         Rules.OdometryDrift);
	// Graphics quality maps to the engine scalability levels (0 = low ... 2 = high).
	const int32 Level = Config.GraphicsQuality == TEXT("high") ? 2 : (Config.GraphicsQuality == TEXT("medium") ? 1 : 0);
	if (GEngine != nullptr)
	{
		GEngine->Exec(World, *FString::Printf(TEXT("sg.ViewDistanceQuality %d"), Level));
		GEngine->Exec(World, *FString::Printf(TEXT("sg.ShadowQuality %d"), Level));
		GEngine->Exec(World, *FString::Printf(TEXT("sg.PostProcessQuality %d"), Level));
	}
	bHasEpisode = true;
	UGameplayStatics::SetGamePaused(this, true);
	return true;
}

void AResearchGameMode::BeginStep(const ResearchProtocol::FAction& Action, int32 Ticks)
{
	check(Character != nullptr);
	Character->SetResearchAction(Action);
	bStepInteract = Action.bInteract;
	StepTicksRemaining = FMath::Clamp(Ticks, 1, ResearchProtocol::MaxTicksPerStep);
	UGameplayStatics::SetGamePaused(this, false);
}

void AResearchGameMode::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (StepTicksRemaining <= 0 || Character == nullptr)
	{
		return;
	}
	UpdateInteraction(bStepInteract);
	++TickCount;
	--StepTicksRemaining;
	UpdateCues();
	if (StepTicksRemaining <= 0 || IsEnded())
	{
		StepTicksRemaining = 0;
		Character->ClearResearchAction();
		UGameplayStatics::SetGamePaused(this, true);
	}
}

AResearchChest* AResearchGameMode::InteractableChest() const
{
	if (Character == nullptr)
	{
		return nullptr;
	}
	const FVector Feet = Character->GetFeetLocation();
	AResearchChest* Best = nullptr;
	double BestDist = TNumericLimits<double>::Max();
	for (AResearchChest* C : Chests)
	{
		if (C == nullptr || !C->IsPresent() || C->IsOpened())
		{
			continue;
		}
		const FVector P = C->GetActorLocation();
		const double Dist = FVector::Dist2D(P, Feet) / 100.0;
		if (Dist > Rules.InteractionRadius || FMath::Abs(P.Z - Feet.Z) / 100.0 > 1.0)
		{
			continue;
		}
		const double Facing = FMath::Abs(AngleDifferenceDeg(BearingDeg(P.X - Feet.X, P.Y - Feet.Y), Character->GetCameraYaw()));
		if (Dist > 0.5 && Facing > Rules.InteractionConeDeg)
		{
			continue;
		}
		FCollisionQueryParams Query(TEXT("ResearchInteractLOS"), true, Character);
		Query.AddIgnoredActor(C);
		FHitResult Hit;
		const FVector Eye = Feet + FVector(0, 0, 100.0);
		if (GetWorld()->LineTraceSingleByChannel(Hit, Eye, C->GetCentre(), ECC_Visibility, Query))
		{
			continue;                                 // something blocks the line of sight
		}
		if (Dist < BestDist)
		{
			BestDist = Dist;
			Best = C;
		}
	}
	return Best;
}

void AResearchGameMode::UpdateInteraction(bool bHeld)
{
	AResearchChest* Target = bHeld ? InteractableChest() : nullptr;
	if (Target == nullptr)
	{
		HoldTarget = INDEX_NONE;
		HoldTicks = 0;
		return;
	}
	if (Target->GetSpawnIndex() != HoldTarget)
	{
		HoldTarget = Target->GetSpawnIndex();
		HoldTicks = 0;
	}
	++HoldTicks;
	const double Held = double(HoldTicks) / Rules.TickRate;
	if (Held + 1e-9 < Rules.InteractionSeconds)
	{
		return;
	}
	HoldTarget = INDEX_NONE;
	HoldTicks = 0;
	const int32 Index = Target->GetSpawnIndex();
	const double Roll = ResearchProtocol::UnitRoll(Config.Seed, Config.Episode, Index, 1 + FailedAttempts);
	const double Now = double(TickCount + 1) / Rules.TickRate;
	if (Roll < Rules.InteractionFailureProbability)
	{
		++FailedAttempts;
		TSharedRef<FJsonObject> E = MakeShared<FJsonObject>();
		E->SetStringField(TEXT("type"), TEXT("interaction_failed"));
		E->SetNumberField(TEXT("sim_time"), Now);
		PendingEvents.Add(MakeShared<FJsonValueObject>(E));
		return;
	}
	Target->SetOpened(true);
	const bool bReward = ResearchProtocol::UnitRoll(Config.Seed, Config.Episode, Index, 999) < Rules.RewardProbability;
	RewardCount += bReward ? 1 : 0;
	RewardFlashUntil.Add(Index, TickCount + Rules.TickRate);
	TSharedRef<FJsonObject> Log = MakeShared<FJsonObject>();
	Log->SetNumberField(TEXT("spawn_index"), Index);
	Log->SetNumberField(TEXT("sim_time"), Now);
	Log->SetNumberField(TEXT("seconds"), Held);
	Log->SetBoolField(TEXT("reward"), bReward);
	OpenedLog.Add(MakeShared<FJsonValueObject>(Log));
	// The in-game "you opened a chest" toast: about the player's own interaction only.
	TSharedRef<FJsonObject> E = MakeShared<FJsonObject>();
	E->SetStringField(TEXT("type"), TEXT("chest_opened_notification"));
	E->SetBoolField(TEXT("reward"), bReward);
	E->SetNumberField(TEXT("sim_time"), Now);
	PendingEvents.Add(MakeShared<FJsonValueObject>(E));
}

void AResearchGameMode::UpdateCues()
{
	AResearchChest* Target = InteractableChest();
	for (AResearchChest* C : Chests)
	{
		if (C == nullptr) { continue; }
		const bool bIsTarget = C == Target;
		C->ShowPrompt(bIsTarget);
		const bool bHolding = bIsTarget && HoldTarget == C->GetSpawnIndex() && HoldTicks > 0;
		C->ShowProgress(bHolding ? float(HoldTicks) / (Rules.InteractionSeconds * Rules.TickRate) : 0.f);
		const int64* Until = RewardFlashUntil.Find(C->GetSpawnIndex());
		C->FlashReward(Until != nullptr && TickCount <= *Until);
	}
}

TSharedRef<FJsonObject> AResearchGameMode::RulesJson() const
{
	const FResearchCharacterParams& P = Character->GetParams();
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetNumberField(TEXT("tick_rate"), Rules.TickRate);
	J->SetNumberField(TEXT("max_ticks"), double(Config.MaxTicks));
	J->SetNumberField(TEXT("interaction_radius"), Rules.InteractionRadius);
	J->SetNumberField(TEXT("interaction_cone_deg"), Rules.InteractionConeDeg);
	J->SetNumberField(TEXT("interaction_seconds"), Rules.InteractionSeconds);
	J->SetNumberField(TEXT("max_speed"), P.MaxSpeed);
	J->SetNumberField(TEXT("sprint_speed"), P.SprintSpeed);
	J->SetNumberField(TEXT("yaw_rate_deg"), P.YawRate);
	J->SetNumberField(TEXT("pitch_rate_deg"), P.PitchRate);
	J->SetNumberField(TEXT("pitch_min_deg"), P.PitchMin);
	J->SetNumberField(TEXT("pitch_max_deg"), P.PitchMax);
	J->SetNumberField(TEXT("capsule_radius"), P.CapsuleRadius);
	J->SetNumberField(TEXT("capsule_height"), P.CapsuleHeight);
	J->SetNumberField(TEXT("step_height"), P.StepHeight);
	return J;
}

TSharedRef<FJsonObject> AResearchGameMode::Telemetry()
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetNumberField(TEXT("sim_time"), GetSimTime());
	J->SetNumberField(TEXT("tick"), double(TickCount));
	J->SetObjectField(TEXT("camera"), Character->CameraStateJson());
	J->SetObjectField(TEXT("odometry"), Character->OdometryJson());
	J->SetArrayField(TEXT("events"), PendingEvents);
	PendingEvents.Reset();
	return J;
}

TSharedRef<FJsonObject> AResearchGameMode::IntrinsicsJson() const
{
	const double Fx = (Config.Width / 2.0) / FMath::Tan(FMath::DegreesToRadians(Config.FovDeg) / 2.0);
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetNumberField(TEXT("width"), Config.Width);
	J->SetNumberField(TEXT("height"), Config.Height);
	J->SetNumberField(TEXT("hfov_deg"), Config.FovDeg);
	J->SetNumberField(TEXT("near"), 0.1);
	J->SetNumberField(TEXT("fx"), Fx);
	J->SetNumberField(TEXT("fy"), Fx);
	J->SetNumberField(TEXT("cx"), Config.Width / 2.0);
	J->SetNumberField(TEXT("cy"), Config.Height / 2.0);
	return J;
}

bool AResearchGameMode::Observe(TSharedPtr<FJsonObject>& OutMeta, TArray<uint8>& OutPixels)
{
	UpdateCues();
	if (!Character->CaptureBgr8(OutPixels))
	{
		return false;
	}
	++FrameId;
	if (bEvaluationEnabled)
	{
		LastLabels = ComputeLabels();
	}
	OutMeta = MakeShared<FJsonObject>();
	OutMeta->SetNumberField(TEXT("frame_id"), double(FrameId));
	OutMeta->SetNumberField(TEXT("sim_time"), GetSimTime());
	OutMeta->SetNumberField(TEXT("width"), Config.Width);
	OutMeta->SetNumberField(TEXT("height"), Config.Height);
	OutMeta->SetStringField(TEXT("encoding"), TEXT("bgr8"));
	OutMeta->SetObjectField(TEXT("intrinsics"), IntrinsicsJson());
	OutMeta->SetObjectField(TEXT("camera"), Character->CameraStateJson());
	OutMeta->SetObjectField(TEXT("odometry"), Character->OdometryJson());
	return true;
}

TSharedRef<FJsonObject> AResearchGameMode::Summary() const
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetNumberField(TEXT("ticks"), double(TickCount));
	J->SetNumberField(TEXT("sim_time"), GetSimTime());
	J->SetNumberField(TEXT("chests_opened"), OpenedLog.Num());
	J->SetNumberField(TEXT("rewards"), RewardCount);
	J->SetNumberField(TEXT("failed_attempts"), FailedAttempts);
	J->SetNumberField(TEXT("travel_distance"), Character != nullptr ? Character->GetDistanceTravelled() : 0.0);
	return J;
}

TSharedRef<FJsonObject> AResearchGameMode::FindPath(const FVector& StartM, const FVector& GoalM, float ToleranceM) const
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	UNavigationSystemV1* Nav = FNavigationSystem::GetCurrent<UNavigationSystemV1>(GetWorld());
	auto Fail = [&J](const TCHAR* Status) {
		J->SetStringField(TEXT("status_code"), Status);
		J->SetArrayField(TEXT("waypoints"), {});
		J->SetNumberField(TEXT("length"), 0.0);
		return J;
	};
	if (Nav == nullptr)
	{
		return Fail(TEXT("unreachable"));
	}
	const FVector Extent(60.0, 60.0, 120.0);
	FNavLocation Start, Goal;
	if (!Nav->ProjectPointToNavigation(ResearchUnits::ToCm(StartM), Start, Extent))
	{
		return Fail(TEXT("start_off_navmesh"));
	}
	const FVector Tol(ToleranceM * 100.0, ToleranceM * 100.0, 120.0);
	if (!Nav->ProjectPointToNavigation(ResearchUnits::ToCm(GoalM), Goal, Tol))
	{
		return Fail(TEXT("goal_off_navmesh"));
	}
	UNavigationPath* Path = UNavigationSystemV1::FindPathToLocationSynchronously(GetWorld(), Start.Location, Goal.Location);
	if (Path == nullptr || !Path->IsValid() || Path->PathPoints.Num() == 0)
	{
		return Fail(TEXT("unreachable"));
	}
	const FVector End = Path->PathPoints.Last();
	if (Path->IsPartial() && FVector::Dist2D(End, Goal.Location) > ToleranceM * 100.0)
	{
		return Fail(TEXT("unreachable"));
	}
	TArray<TSharedPtr<FJsonValue>> Points;
	double Length = 0.0;
	for (int32 I = 0; I < Path->PathPoints.Num(); ++I)
	{
		const FVector M = ResearchUnits::ToMetres(Path->PathPoints[I]);
		Points.Add(MakeShared<FJsonValueArray>(Vec(M)));
		if (I > 0) { Length += FVector::Dist(Path->PathPoints[I], Path->PathPoints[I - 1]) / 100.0; }
	}
	J->SetStringField(TEXT("status_code"), TEXT("ok"));
	J->SetArrayField(TEXT("waypoints"), Points);
	J->SetNumberField(TEXT("length"), Length);
	return J;
}

float AResearchGameMode::NavRaycast(const FVector& OriginM, float BearingDegIn, float MaxDistanceM) const
{
	UNavigationSystemV1* Nav = FNavigationSystem::GetCurrent<UNavigationSystemV1>(GetWorld());
	if (Nav == nullptr)
	{
		return 0.f;
	}
	FNavLocation Origin;
	if (!Nav->ProjectPointToNavigation(ResearchUnits::ToCm(OriginM), Origin, FVector(60.0, 60.0, 120.0)))
	{
		return 0.f;
	}
	const double Rad = FMath::DegreesToRadians(BearingDegIn);
	const FVector End = Origin.Location + FVector(FMath::Cos(Rad), FMath::Sin(Rad), 0.0) * MaxDistanceM * 100.0;
	FVector Hit;
	if (UNavigationSystemV1::NavigationRaycast(GetWorld(), Origin.Location, End, Hit))
	{
		return float(FVector::Dist2D(Origin.Location, Hit) / 100.0);
	}
	return MaxDistanceM;
}

TSharedRef<FJsonObject> AResearchGameMode::OracleState() const
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	TSharedRef<FJsonObject> Agent = MakeShared<FJsonObject>();
	Agent->SetArrayField(TEXT("position"), Vec(ResearchUnits::ToMetres(Character->GetFeetLocation())));
	Agent->SetNumberField(TEXT("camera_yaw_deg"), Character->GetCameraYaw());
	J->SetObjectField(TEXT("agent"), Agent);
	TArray<TSharedPtr<FJsonValue>> List;
	for (const AResearchChest* C : Chests)
	{
		TSharedRef<FJsonObject> O = MakeShared<FJsonObject>();
		O->SetNumberField(TEXT("spawn_index"), C->GetSpawnIndex());
		O->SetStringField(TEXT("name"), C->GetSpawnName());
		O->SetArrayField(TEXT("position"), Vec(ResearchUnits::ToMetres(C->GetActorLocation())));
		O->SetBoolField(TEXT("present"), C->IsPresent());
		O->SetBoolField(TEXT("opened"), C->IsOpened());
		List.Add(MakeShared<FJsonValueObject>(O));
	}
	J->SetArrayField(TEXT("chests"), List);
	return J;
}

TSharedPtr<FJsonObject> AResearchGameMode::ComputeLabels() const
{
	// Evaluation labels for the frame just captured. Unlike the reference renderer (exact
	// per-pixel ids) these are APPROXIMATE: projected bounding boxes of each chest, kept when
	// at least one of the chest's sample points is visible from the camera (line traces).
	const FTransform View = Character->GetViewTransform();
	const FVector Cam = View.GetLocation();
	const FRotator Rot = View.Rotator();
	const FVector F = Rot.Vector();
	const FVector R = FRotationMatrix(Rot).GetUnitAxis(EAxis::Y);
	const FVector U = FRotationMatrix(Rot).GetUnitAxis(EAxis::Z);
	const double Fx = (Config.Width / 2.0) / FMath::Tan(FMath::DegreesToRadians(Config.FovDeg) / 2.0);
	const double Cx = Config.Width / 2.0, Cy = Config.Height / 2.0;
	TArray<TSharedPtr<FJsonValue>> Labels;
	for (const AResearchChest* C : Chests)
	{
		if (!C->IsPresent()) { continue; }
		const FBox Box = C->GetComponentsBoundingBox(false);
		double U0 = TNumericLimits<double>::Max(), V0 = U0, U1 = -U0, V1 = -U0;
		bool bInFront = false;
		for (int32 K = 0; K < 8; ++K)
		{
			const FVector P((K & 1) ? Box.Max.X : Box.Min.X, (K & 2) ? Box.Max.Y : Box.Min.Y, (K & 4) ? Box.Max.Z : Box.Min.Z);
			const FVector D = P - Cam;
			const double Depth = FVector::DotProduct(D, F);
			if (Depth < 10.0) { continue; }
			bInFront = true;
			const double Px = Cx + Fx * FVector::DotProduct(D, R) / Depth;
			const double Py = Cy - Fx * FVector::DotProduct(D, U) / Depth;
			U0 = FMath::Min(U0, Px); U1 = FMath::Max(U1, Px); V0 = FMath::Min(V0, Py); V1 = FMath::Max(V1, Py);
		}
		if (!bInFront) { continue; }
		U0 = FMath::Clamp(U0, 0.0, Config.Width - 1.0); U1 = FMath::Clamp(U1, 0.0, Config.Width - 1.0);
		V0 = FMath::Clamp(V0, 0.0, Config.Height - 1.0); V1 = FMath::Clamp(V1, 0.0, Config.Height - 1.0);
		if (U1 - U0 < 1.0 || V1 - V0 < 1.0) { continue; }
		bool bVisible = false;
		FCollisionQueryParams Query(TEXT("ResearchLabelLOS"), true, Character);
		Query.AddIgnoredActor(C);
		for (const FVector& Sample : { Box.GetCenter(), FVector(Box.GetCenter().X, Box.GetCenter().Y, Box.Max.Z - 2.0) })
		{
			FHitResult Hit;
			if (!GetWorld()->LineTraceSingleByChannel(Hit, Cam, Sample, ECC_Visibility, Query)) { bVisible = true; break; }
		}
		if (!bVisible) { continue; }
		TSharedRef<FJsonObject> O = MakeShared<FJsonObject>();
		O->SetNumberField(TEXT("spawn_index"), C->GetSpawnIndex());
		O->SetStringField(TEXT("category"), C->IsOpened() ? TEXT("chest_opened") : TEXT("chest"));
		O->SetArrayField(TEXT("bbox"), { MakeShared<FJsonValueNumber>(FMath::FloorToInt(U0)), MakeShared<FJsonValueNumber>(FMath::FloorToInt(V0)),
		                                 MakeShared<FJsonValueNumber>(FMath::CeilToInt(U1 - U0)), MakeShared<FJsonValueNumber>(FMath::CeilToInt(V1 - V0)) });
		O->SetNumberField(TEXT("pixels"), FMath::RoundToInt((U1 - U0) * (V1 - V0)));
		O->SetArrayField(TEXT("position"), Vec(ResearchUnits::ToMetres(C->GetActorLocation())));
		O->SetNumberField(TEXT("distance"), FVector::Dist(Cam, C->GetCentre()) / 100.0);
		Labels.Add(MakeShared<FJsonValueObject>(O));
	}
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetNumberField(TEXT("frame_id"), double(FrameId));
	J->SetArrayField(TEXT("chests"), Labels);
	const FVector CamM = ResearchUnits::ToMetres(Cam);
	J->SetArrayField(TEXT("camera_pose"), { MakeShared<FJsonValueNumber>(CamM.X), MakeShared<FJsonValueNumber>(CamM.Y),
	                                        MakeShared<FJsonValueNumber>(CamM.Z), MakeShared<FJsonValueNumber>(Rot.Yaw),
	                                        MakeShared<FJsonValueNumber>(Rot.Pitch) });
	J->SetArrayField(TEXT("agent_position"), Vec(ResearchUnits::ToMetres(Character->GetFeetLocation())));
	J->SetNumberField(TEXT("agent_pixels"), -1);       // not measured in Unreal (no id buffer)
	J->SetBoolField(TEXT("approximate"), true);
	return J;
}
