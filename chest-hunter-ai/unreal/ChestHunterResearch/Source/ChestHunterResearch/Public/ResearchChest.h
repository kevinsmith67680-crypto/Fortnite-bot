// A research chest: closed/opened geometry built from engine basic shapes, plus the
// in-world interaction cues (prompt sign, hold-progress bar, reward flash) that the camera
// can see. Presence and opened state are ground truth: they are never sent to the agent;
// the agent only ever receives camera pixels.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ResearchChest.generated.h"

class UStaticMeshComponent;
class UMaterialInstanceDynamic;

UCLASS()
class CHESTHUNTERRESEARCH_API AResearchChest : public AActor
{
	GENERATED_BODY()

public:
	AResearchChest();

	/** Dimensions in metres (must match chest_hunter.simulation3d.arena constants). */
	static constexpr float WidthM = 0.9f;
	static constexpr float DepthM = 0.6f;
	static constexpr float BodyHeightM = 0.5f;
	static constexpr float LidHeightM = 0.15f;

	/** ``ValueScale``/``HueShiftDeg``: procedural chest appearance (1 and 0 = standard). */
	void Initialise(int32 InSpawnIndex, const FString& InName, float InYawDeg, bool bInPresent,
	                float ValueScale = 1.f, float HueShiftDeg = 0.f);
	void SetOpened(bool bInOpened);
	/** Interaction cues; Fraction in [0, 1] (<= 0 hides the bar). */
	void ShowPrompt(bool bVisible);
	void ShowProgress(float Fraction);
	void FlashReward(bool bVisible);

	int32 GetSpawnIndex() const { return SpawnIndex; }
	const FString& GetSpawnName() const { return SpawnName; }
	bool IsPresent() const { return bPresent; }
	bool IsOpened() const { return bOpened; }
	/** World-space centre of the chest (cm). */
	FVector GetCentre() const;

private:
	UStaticMeshComponent* MakePart(const TCHAR* Name, const FLinearColor& Colour);
	void PlacePart(UStaticMeshComponent* Part, const FVector& CentreM, const FVector& SizeM);
	void LayoutClosed();
	void LayoutOpened();
	void ApplyVisibility();

	UPROPERTY(VisibleAnywhere) TObjectPtr<USceneComponent> Root;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> Body;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> Lid;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> Latch;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> OpenBody;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> Inside;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> OpenLid;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> PromptSign;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> ProgressFrame;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> ProgressFill;
	UPROPERTY(VisibleAnywhere) TObjectPtr<UStaticMeshComponent> RewardFlash;

	int32 SpawnIndex = -1;
	FString SpawnName;
	float YawDeg = 0.f;
	bool bPresent = false;
	bool bOpened = false;
};
