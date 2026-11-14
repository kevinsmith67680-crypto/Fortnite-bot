#include "ResearchCharacter.h"

#include "ChestHunterResearch.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "TextureResource.h"
#include "UObject/ConstructorHelpers.h"

AResearchCharacter::AResearchCharacter()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bTickEvenWhenPaused = false;

	const float Cm = 100.f;
	GetCapsuleComponent()->InitCapsuleSize(Params.CapsuleRadius * Cm, Params.CapsuleHeight * Cm / 2);
	bUseControllerRotationYaw = false;
	bUseControllerRotationPitch = false;
	bUseControllerRotationRoll = false;

	UCharacterMovementComponent* Move = GetCharacterMovement();
	Move->MaxWalkSpeed = Params.MaxSpeed * Cm;
	Move->MaxWalkSpeedCrouched = Params.CrouchSpeed * Cm;
	Move->MaxAcceleration = 2000.f;
	Move->BrakingDecelerationWalking = 3000.f;
	Move->JumpZVelocity = Params.JumpSpeed * Cm;
	Move->MaxStepHeight = Params.StepHeight * Cm;
	Move->bOrientRotationToMovement = true;
	Move->RotationRate = FRotator(0.f, 540.f, 0.f);
	Move->GetNavAgentPropertiesRef().bCanCrouch = true;
	Move->GetNavAgentPropertiesRef().AgentRadius = Params.CapsuleRadius * Cm;
	Move->GetNavAgentPropertiesRef().AgentHeight = Params.CapsuleHeight * Cm;

	// Visible body (engine basic shapes; colours from geometry.COLOURS agent_*).
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
	auto MakeBodyPart = [&](const TCHAR* Name, const FVector& Offset, const FVector& Size) {
		UStaticMeshComponent* Part = CreateDefaultSubobject<UStaticMeshComponent>(Name);
		Part->SetupAttachment(GetCapsuleComponent());
		if (Cube.Succeeded()) { Part->SetStaticMesh(Cube.Object); }
		Part->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		Part->SetRelativeLocation(Offset * Cm);
		Part->SetRelativeScale3D(Size);
		return Part;
	};
	const float HalfH = Params.CapsuleHeight / 2;       // capsule origin is at mid height
	BodyMesh = MakeBodyPart(TEXT("BodyMesh"), FVector(0, 0, 0.05f + 0.6f - HalfH), FVector(0.4f, 0.56f, 1.2f));
	HeadMesh = MakeBodyPart(TEXT("HeadMesh"), FVector(0, 0, 1.27f + 0.16f - HalfH), FVector(0.32f, 0.32f, 0.32f));
	VisorMesh = MakeBodyPart(TEXT("VisorMesh"), FVector(0.17f, 0, 1.4f + 0.05f - HalfH), FVector(0.06f, 0.24f, 0.1f));

	// Over-the-shoulder spring arm. Its pivot is PivotHeight above the feet.
	SpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("SpringArm"));
	SpringArm->SetupAttachment(GetCapsuleComponent());
	SpringArm->SetRelativeLocation(FVector(0, 0, (Params.PivotHeight - HalfH) * Cm));
	SpringArm->TargetArmLength = Params.ArmLength * Cm;
	SpringArm->SocketOffset = FVector(0, Params.ShoulderOffset * Cm, 0);
	SpringArm->bDoCollisionTest = true;
	SpringArm->ProbeSize = 20.f;
	SpringArm->bUsePawnControlRotation = false;     // the bridge sets the arm rotation directly
	SpringArm->bInheritYaw = false;
	SpringArm->bInheritPitch = false;
	SpringArm->bInheritRoll = false;
	SpringArm->bEnableCameraLag = false;

	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(SpringArm, USpringArmComponent::SocketName);

	// The agent's camera: a scene capture at the follow camera, rendered on demand only.
	Capture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("AgentCapture"));
	Capture->SetupAttachment(FollowCamera);
	Capture->bCaptureEveryFrame = false;
	Capture->bCaptureOnMovement = false;
	Capture->bAlwaysPersistRenderingState = true;
	Capture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	// No editor-only or debug visualisation in the agent's image.
	Capture->ShowFlags.SetBillboardSprites(false);
	Capture->ShowFlags.SetNavigation(false);
	Capture->ShowFlags.SetCollision(false);
	Capture->ShowFlags.SetBounds(false);
}

void AResearchCharacter::BeginPlay()
{
	Super::BeginPlay();
	// Colour the body parts like the reference arena's agent.
	if (UMaterialInterface* Base = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")))
	{
		const TPair<UStaticMeshComponent*, FColor> Parts[] = {
			{ BodyMesh, FColor(59, 130, 246) }, { HeadMesh, FColor(220, 180, 150) }, { VisorMesh, FColor(255, 255, 255) } };
		for (const TPair<UStaticMeshComponent*, FColor>& P : Parts)
		{
			UMaterialInstanceDynamic* Mid = UMaterialInstanceDynamic::Create(Base, this);
			Mid->SetVectorParameterValue(TEXT("Color"), FLinearColor(P.Value));
			P.Key->SetMaterial(0, Mid);
		}
	}
	LastFeetCm = GetFeetLocation();
}

FVector AResearchCharacter::GetFeetLocation() const
{
	return GetActorLocation() - FVector(0, 0, GetCapsuleComponent()->GetScaledCapsuleHalfHeight());
}

void AResearchCharacter::ResetTo(const FVector& FeetCm, float YawDeg)
{
	const FVector Centre = FeetCm + FVector(0, 0, GetCapsuleComponent()->GetScaledCapsuleHalfHeight() + 2.f);
	SetActorLocationAndRotation(Centre, FRotator(0.f, YawDeg, 0.f), false, nullptr, ETeleportType::ResetPhysics);
	GetCharacterMovement()->StopMovementImmediately();
	CameraYaw = FRotator::NormalizeAxis(YawDeg);
	CameraPitch = -12.f;
	SpringArm->SetWorldRotation(FRotator(CameraPitch, CameraYaw, 0.f));
	PendingAction = ResearchProtocol::FAction();
	LastFeetCm = GetFeetLocation();
	DistanceTravelledM = 0.0;
	OdometryDriftM = FVector2D::ZeroVector;
}

void AResearchCharacter::ConfigureCamera(int32 Width, int32 Height, float HorizontalFovDeg)
{
	if (RenderTarget == nullptr || RenderTarget->SizeX != Width || RenderTarget->SizeY != Height)
	{
		RenderTarget = NewObject<UTextureRenderTarget2D>(this);
		RenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
		RenderTarget->bAutoGenerateMips = false;
		RenderTarget->ClearColor = FLinearColor::Black;
		RenderTarget->InitAutoFormat(Width, Height);
		RenderTarget->UpdateResourceImmediate(true);
	}
	Capture->TextureTarget = RenderTarget;
	Capture->FOVAngle = HorizontalFovDeg;          // horizontal FOV, as in CameraIntrinsics
	FollowCamera->SetFieldOfView(HorizontalFovDeg);
}

bool AResearchCharacter::CaptureBgr8(TArray<uint8>& OutPixels)
{
	if (RenderTarget == nullptr)
	{
		return false;
	}
	Capture->CaptureScene();                         // enqueues the render of the agent's view
	FTextureRenderTargetResource* Resource = RenderTarget->GameThread_GetRenderTargetResource();
	if (Resource == nullptr)
	{
		return false;
	}
	TArray<FColor> Pixels;
	FReadSurfaceDataFlags Flags(RCM_UNorm);
	Flags.SetLinearToGamma(false);
	if (!Resource->ReadPixels(Pixels, Flags))          // flushes rendering commands (synchronous)
	{
		return false;
	}
	const int32 Count = RenderTarget->SizeX * RenderTarget->SizeY;
	if (Pixels.Num() != Count)
	{
		return false;
	}
	OutPixels.SetNumUninitialized(Count * 3);
	uint8* Out = OutPixels.GetData();
	for (int32 I = 0; I < Count; ++I)                // FColor is stored B, G, R, A
	{
		Out[3 * I + 0] = Pixels[I].B;
		Out[3 * I + 1] = Pixels[I].G;
		Out[3 * I + 2] = Pixels[I].R;
	}
	return true;
}

FTransform AResearchCharacter::GetViewTransform() const
{
	return FollowCamera->GetComponentTransform();
}

void AResearchCharacter::ApplyResearchAction(const ResearchProtocol::FAction& Action, float DeltaSeconds)
{
	// Look: rates are fractions of the configured maximum turn rate (protocol contract).
	CameraYaw = FRotator::NormalizeAxis(CameraYaw + Action.LookYaw * Params.YawRate * DeltaSeconds);
	CameraPitch = FMath::Clamp(CameraPitch + Action.LookPitch * Params.PitchRate * DeltaSeconds,
	                           Params.PitchMin, Params.PitchMax);
	SpringArm->SetWorldRotation(FRotator(CameraPitch, CameraYaw, 0.f));

	// Move relative to the camera yaw (forward = camera bearing), like the reference arena.
	UCharacterMovementComponent* Move = GetCharacterMovement();
	Move->MaxWalkSpeed = (Action.bSprint ? Params.SprintSpeed : Params.MaxSpeed) * 100.f;
	const FRotator YawOnly(0.f, CameraYaw, 0.f);
	const FVector Forward = YawOnly.Vector();
	const FVector Right = FRotationMatrix(YawOnly).GetUnitAxis(EAxis::Y);
	FVector Input = Forward * Action.MoveForward + Right * Action.MoveRight;
	if (Input.SizeSquared() > 1.0)
	{
		Input.Normalize();
	}
	AddMovementInput(Input, 1.0f);
	if (Action.bJump) { Jump(); } else { StopJumping(); }
	if (Action.bCrouch) { Crouch(); } else { UnCrouch(); }
}

void AResearchCharacter::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	// Runs before the movement component (which has this pawn as a tick prerequisite).
	ApplyResearchAction(PendingAction, DeltaSeconds);
	// Jump is a one-tick impulse in the protocol: clear it after it has been applied once.
	PendingAction.bJump = false;
	UpdateOdometry();
}

void AResearchCharacter::ResetOdometry(uint64 Seed, double InDriftPerMetre)
{
	OdometryState = Seed == 0 ? 0x9E3779B97F4A7C15ull : Seed;
	DriftPerMetre = InDriftPerMetre;
	OdometryDriftM = FVector2D::ZeroVector;
}

double AResearchCharacter::NextGaussian()
{
	// Box-Muller on a SplitMix64 stream (seeded per episode; independent of frame rate).
	OdometryState = ResearchProtocol::SplitMix64(OdometryState);
	const double U1 = FMath::Max(double(OdometryState >> 11) / double(1ull << 53), 1e-12);
	OdometryState = ResearchProtocol::SplitMix64(OdometryState);
	const double U2 = double(OdometryState >> 11) / double(1ull << 53);
	return FMath::Sqrt(-2.0 * FMath::Loge(U1)) * FMath::Cos(2.0 * PI * U2);
}

void AResearchCharacter::UpdateOdometry()
{
	const FVector Feet = GetFeetLocation();
	const double MovedM = FVector::Dist2D(Feet, LastFeetCm) / 100.0;
	LastFeetCm = Feet;
	DistanceTravelledM += MovedM;
	if (MovedM > 0.0 && DriftPerMetre > 0.0)
	{
		OdometryDriftM.X += NextGaussian() * DriftPerMetre * MovedM;
		OdometryDriftM.Y += NextGaussian() * DriftPerMetre * MovedM;
	}
}

TSharedRef<FJsonObject> AResearchCharacter::CameraStateJson() const
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	const FVector Cam = FollowCamera->GetComponentLocation();
	const FVector Rel = ResearchUnits::ToMetres(Cam - GetFeetLocation());
	J->SetNumberField(TEXT("yaw_deg"), FMath::Fmod(CameraYaw + 360.0, 360.0));
	J->SetNumberField(TEXT("pitch_deg"), CameraPitch);
	J->SetNumberField(TEXT("arm_length"), FVector::Dist(Cam, SpringArm->GetComponentLocation()) / 100.0);
	TArray<TSharedPtr<FJsonValue>> P = { MakeShared<FJsonValueNumber>(Rel.X), MakeShared<FJsonValueNumber>(Rel.Y),
	                                     MakeShared<FJsonValueNumber>(Rel.Z) };
	J->SetArrayField(TEXT("relative_position"), P);
	return J;
}

TSharedRef<FJsonObject> AResearchCharacter::OdometryJson() const
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	const FVector Feet = ResearchUnits::ToMetres(GetFeetLocation());
	const FVector Vel = ResearchUnits::ToMetres(GetVelocity());
	J->SetArrayField(TEXT("position"), { MakeShared<FJsonValueNumber>(Feet.X + OdometryDriftM.X),
	                                     MakeShared<FJsonValueNumber>(Feet.Y + OdometryDriftM.Y),
	                                     MakeShared<FJsonValueNumber>(Feet.Z) });
	J->SetArrayField(TEXT("velocity"), { MakeShared<FJsonValueNumber>(Vel.X), MakeShared<FJsonValueNumber>(Vel.Y),
	                                     MakeShared<FJsonValueNumber>(Vel.Z) });
	J->SetBoolField(TEXT("grounded"), GetCharacterMovement()->IsMovingOnGround());
	return J;
}
