#include "ResearchChest.h"

#include "ChestHunterResearch.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"

namespace
{
	// Colours mirror chest_hunter.simulation3d.geometry.COLOURS (given there as BGR).
	FLinearColor Bgr(uint8 B, uint8 G, uint8 R) { return FLinearColor(FColor(R, G, B)); }
	const FLinearColor ChestBody = Bgr(24, 197, 245);
	const FLinearColor ChestLid = Bgr(20, 72, 120);
	const FLinearColor ChestLatch = Bgr(40, 40, 44);
	const FLinearColor ChestInside = Bgr(160, 210, 230);
	const FLinearColor ChestOpenBody = Bgr(30, 150, 200);
	const FLinearColor PromptColour = Bgr(255, 255, 255);
	const FLinearColor ProgressFrameColour = Bgr(255, 255, 255);
	const FLinearColor ProgressFillColour = Bgr(94, 197, 34);
	const FLinearColor RewardColour = Bgr(238, 211, 34);
	const TCHAR* CubePath = TEXT("/Engine/BasicShapes/Cube.Cube");               // 100 cm cube
	const TCHAR* MaterialPath = TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial");
}

AResearchChest::AResearchChest()
{
	PrimaryActorTick.bCanEverTick = false;
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
	Body = MakePart(TEXT("Body"), ChestBody);
	Lid = MakePart(TEXT("Lid"), ChestLid);
	Latch = MakePart(TEXT("Latch"), ChestLatch);
	OpenBody = MakePart(TEXT("OpenBody"), ChestOpenBody);
	Inside = MakePart(TEXT("Inside"), ChestInside);
	OpenLid = MakePart(TEXT("OpenLid"), ChestLid);
	PromptSign = MakePart(TEXT("PromptSign"), PromptColour);
	ProgressFrame = MakePart(TEXT("ProgressFrame"), ProgressFrameColour);
	ProgressFill = MakePart(TEXT("ProgressFill"), ProgressFillColour);
	RewardFlash = MakePart(TEXT("RewardFlash"), RewardColour);
	for (UStaticMeshComponent* Cue : { PromptSign.Get(), ProgressFrame.Get(), ProgressFill.Get(), RewardFlash.Get() })
	{
		Cue->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		Cue->SetCastShadow(false);
		Cue->SetCanEverAffectNavigation(false);
	}
}

UStaticMeshComponent* AResearchChest::MakePart(const TCHAR* Name, const FLinearColor& Colour)
{
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(CubePath);
	UStaticMeshComponent* Part = CreateDefaultSubobject<UStaticMeshComponent>(Name);
	Part->SetupAttachment(Root);
	if (Cube.Succeeded())
	{
		Part->SetStaticMesh(Cube.Object);
	}
	Part->SetCollisionProfileName(TEXT("BlockAll"));
	Part->ComponentTags.Add(FName(*FString::Printf(TEXT("Colour:%s"), *Colour.ToFColor(true).ToHex())));
	return Part;
}

void AResearchChest::PlacePart(UStaticMeshComponent* Part, const FVector& CentreM, const FVector& SizeM)
{
	// Engine cube is 100 cm centred on its pivot.
	Part->SetRelativeLocation(ResearchUnits::ToCm(CentreM));
	Part->SetRelativeScale3D(SizeM);
}

void AResearchChest::Initialise(int32 InSpawnIndex, const FString& InName, float InYawDeg, bool bInPresent,
                                float ValueScale, float HueShiftDeg)
{
	SpawnIndex = InSpawnIndex;
	SpawnName = InName;
	YawDeg = InYawDeg;
	bPresent = bInPresent;
	bOpened = false;
	// Local frame: +X is the chest front; the actor is rotated to face YawDeg.
	SetActorRotation(FRotator(0.f, YawDeg, 0.f));
	UObject* MaterialObject = StaticLoadObject(UMaterialInterface::StaticClass(), nullptr, MaterialPath);
	UMaterialInterface* Base = Cast<UMaterialInterface>(MaterialObject);
	const TPair<UStaticMeshComponent*, FLinearColor> Parts[] = {
		{ Body, ChestBody }, { Lid, ChestLid }, { Latch, ChestLatch }, { OpenBody, ChestOpenBody },
		{ Inside, ChestInside }, { OpenLid, ChestLid }, { PromptSign, PromptColour },
		{ ProgressFrame, ProgressFrameColour }, { ProgressFill, ProgressFillColour }, { RewardFlash, RewardColour } };
	if (Base != nullptr)
	{
		for (const TPair<UStaticMeshComponent*, FLinearColor>& P : Parts)
		{
			FLinearColor Colour = P.Value;
			const bool bChestPart = P.Key != PromptSign && P.Key != ProgressFrame
				&& P.Key != ProgressFill && P.Key != RewardFlash;
			if (bChestPart && (ValueScale != 1.f || HueShiftDeg != 0.f))
			{
				FLinearColor Hsv = Colour.LinearRGBToHSV();   // R = hue (deg), G = sat, B = value
				Hsv.R = FMath::Fmod(Hsv.R + HueShiftDeg + 360.f, 360.f);
				Hsv.B = FMath::Clamp(Hsv.B * ValueScale, 0.f, 1.f);
				Colour = Hsv.HSVToLinearRGB();
			}
			UMaterialInstanceDynamic* Mid = UMaterialInstanceDynamic::Create(Base, this);
			Mid->SetVectorParameterValue(TEXT("Color"), Colour);
			P.Key->SetMaterial(0, Mid);
		}
	}
	else
	{
		UE_LOG(LogChestHunter, Warning, TEXT("BasicShapeMaterial not found; chests use the default material"));
	}
	LayoutClosed();
	LayoutOpened();
	ShowPrompt(false);
	ShowProgress(0.f);
	FlashReward(false);
	ApplyVisibility();
}

void AResearchChest::LayoutClosed()
{
	const float D = DepthM, W = WidthM, H = BodyHeightM, L = LidHeightM;
	PlacePart(Body, FVector(0, 0, H / 2), FVector(D, W, H));
	PlacePart(Lid, FVector(0, 0, H + L / 2), FVector(D, W, L));
	PlacePart(Latch, FVector(D / 2 + 0.02f, 0, 0.40f), FVector(0.06f, 0.16f, 0.16f));
}

void AResearchChest::LayoutOpened()
{
	const float D = DepthM, W = WidthM, H = BodyHeightM;
	PlacePart(OpenBody, FVector(0, 0, (H - 0.02f) / 2), FVector(D, W, H - 0.02f));
	PlacePart(Inside, FVector(0, 0, H - 0.01f), FVector(D - 0.1f, W - 0.1f, 0.02f));
	PlacePart(OpenLid, FVector(-(D / 2 - 0.06f), 0, H + 0.28f), FVector(0.12f, W, 0.56f));
	// Cues float above the chest (the prompt faces every direction: it is a thin slab).
	PlacePart(PromptSign, FVector(0, 0, H + LidHeightM + 0.55f), FVector(0.05f, 0.64f, 0.32f));
	PlacePart(ProgressFrame, FVector(0, 0, H + LidHeightM + 0.2f), FVector(0.04f, 0.8f, 0.1f));
	PlacePart(RewardFlash, FVector(0, 0, 1.6f), FVector(0.35f, 0.35f, 0.35f));
}

void AResearchChest::ApplyVisibility()
{
	const bool bClosed = bPresent && !bOpened;
	const bool bOpen = bPresent && bOpened;
	for (UStaticMeshComponent* Part : { Body.Get(), Lid.Get(), Latch.Get() })
	{
		Part->SetVisibility(bClosed);
		Part->SetCollisionEnabled(bClosed ? ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
	}
	for (UStaticMeshComponent* Part : { OpenBody.Get(), Inside.Get(), OpenLid.Get() })
	{
		Part->SetVisibility(bOpen);
		Part->SetCollisionEnabled(bOpen ? ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
	}
	// Chests are dynamic navigation obstacles, exactly like the reference arena.
	SetActorEnableCollision(bPresent);
}

void AResearchChest::SetOpened(bool bInOpened)
{
	bOpened = bInOpened;
	ApplyVisibility();
}

void AResearchChest::ShowPrompt(bool bVisible)
{
	PromptSign->SetVisibility(bVisible && bPresent && !bOpened);
}

void AResearchChest::ShowProgress(float Fraction)
{
	const bool bVisible = Fraction > 0.f && bPresent && !bOpened;
	ProgressFrame->SetVisibility(bVisible);
	ProgressFill->SetVisibility(bVisible);
	if (bVisible)
	{
		const float F = FMath::Clamp(Fraction, 0.f, 1.f);
		const float Bar = 0.76f * F;
		const float Z = BodyHeightM + LidHeightM + 0.2f;
		// The fill grows from the left edge of the frame and sits just in front of it.
		PlacePart(ProgressFill, FVector(0.03f, -0.38f + Bar / 2, Z), FVector(0.04f, FMath::Max(Bar, 0.001f), 0.07f));
	}
}

void AResearchChest::FlashReward(bool bVisible)
{
	RewardFlash->SetVisibility(bVisible && bPresent);
}

FVector AResearchChest::GetCentre() const
{
	return GetActorLocation() + FVector(0, 0, (BodyHeightM + LidHeightM) * 50.0);
}
