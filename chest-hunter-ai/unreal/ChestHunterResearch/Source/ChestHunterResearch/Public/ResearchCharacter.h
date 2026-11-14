// Third-person research character. Movement and camera are driven ONLY by validated bridge
// actions (ApplyResearchAction); there is no operating-system input injection. The agent's
// camera image comes from a USceneCaptureComponent2D attached to the follow camera, i.e. the
// game's own third-person view rendered into a render target (never a desktop screenshot).

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "ResearchProtocol.h"
#include "ResearchCharacter.generated.h"

class USpringArmComponent;
class UCameraComponent;
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;
class UStaticMeshComponent;

/** Must match chest_hunter.simulation3d.character.CharacterParams. */
USTRUCT()
struct FResearchCharacterParams
{
	GENERATED_BODY()
	float MaxSpeed = 5.0f;          // m/s
	float SprintSpeed = 7.0f;
	float CrouchSpeed = 2.5f;
	float JumpSpeed = 4.2f;
	float YawRate = 120.f;          // deg/s at look_yaw = 1
	float PitchRate = 60.f;         // deg/s at look_pitch = 1
	float PitchMin = -55.f;
	float PitchMax = 25.f;
	float PivotHeight = 1.6f;       // camera pivot above the feet (m)
	float ArmLength = 3.2f;
	float ShoulderOffset = 0.65f;
	float CapsuleRadius = 0.35f;
	float CapsuleHeight = 1.8f;
	float StepHeight = 0.55f;
};

UCLASS()
class CHESTHUNTERRESEARCH_API AResearchCharacter : public ACharacter
{
	GENERATED_BODY()

public:
	AResearchCharacter();

	/** Queue an action that is applied on every world tick until replaced. */
	void SetResearchAction(const ResearchProtocol::FAction& Action) { PendingAction = Action; }
	void ClearResearchAction() { PendingAction = ResearchProtocol::FAction(); }

	/** Place the character at an episode start (feet position in cm, yaw in degrees). */
	void ResetTo(const FVector& FeetCm, float YawDeg);

	/** Configure and (re)allocate the capture target. */
	void ConfigureCamera(int32 Width, int32 Height, float HorizontalFovDeg);
	/** Render the agent's view now and read it back as tightly packed bgr8 rows. */
	bool CaptureBgr8(TArray<uint8>& OutPixels);

	/** Agent-facing camera rig state (orientation + position relative to the feet, metres). */
	TSharedRef<FJsonObject> CameraStateJson() const;
	/** Odometry with seeded random-walk drift (the agent never sees the true position). */
	TSharedRef<FJsonObject> OdometryJson() const;
	void ResetOdometry(uint64 Seed, double DriftPerMetre);

	FVector GetFeetLocation() const;
	float GetCameraYaw() const { return CameraYaw; }
	float GetCameraPitch() const { return CameraPitch; }
	double GetDistanceTravelled() const { return DistanceTravelledM; }
	const FResearchCharacterParams& GetParams() const { return Params; }
	USceneCaptureComponent2D* GetCapture() const { return Capture; }
	FTransform GetViewTransform() const;

	virtual void Tick(float DeltaSeconds) override;

protected:
	virtual void BeginPlay() override;

private:
	void ApplyResearchAction(const ResearchProtocol::FAction& Action, float DeltaSeconds);
	void UpdateOdometry();
	double NextGaussian();

	UPROPERTY(VisibleAnywhere) TObjectPtr<USpringArmComponent> SpringArm;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UCameraComponent> FollowCamera;
	UPROPERTY(VisibleAnywhere) TObjectPtr<USceneCaptureComponent2D> Capture;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> BodyMesh;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> HeadMesh;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> VisorMesh;
	UPROPERTY(Transient) TObjectPtr<UTextureRenderTarget2D> RenderTarget;

	FResearchCharacterParams Params;
	ResearchProtocol::FAction PendingAction;
	float CameraYaw = 0.f;
	float CameraPitch = -12.f;
	FVector LastFeetCm = FVector::ZeroVector;
	double DistanceTravelledM = 0.0;
	FVector2D OdometryDriftM = FVector2D::ZeroVector;
	double DriftPerMetre = 0.005;
	uint64 OdometryState = 1;
};
