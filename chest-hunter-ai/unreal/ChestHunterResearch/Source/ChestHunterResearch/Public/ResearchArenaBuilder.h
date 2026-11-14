// Builds the research arena at runtime from Content/ResearchData/<map_id>.json, the layout
// exported by `python -m chest_hunter export-arena-3d` (the same data the Python reference
// arena uses). Geometry is made of engine cubes, so the project needs no binary art assets.

#pragma once

#include "CoreMinimal.h"
#include "ResearchArenaBuilder.generated.h"

class AStaticMeshActor;

USTRUCT()
struct FResearchSpawnPoint
{
	GENERATED_BODY()
	int32 Index = 0;
	FString Region;
	FString Name;
	FString LocationName;
	FVector PositionM = FVector::ZeroVector;    // floor position the chest stands on (metres)
	float YawDeg = 0.f;
	float Probability = 0.f;
	bool bReachable = true;
};

USTRUCT()
struct FResearchArenaLayout
{
	GENERATED_BODY()
	FString MapId;
	FBox2D BoundsM = FBox2D(ForceInit);
	FVector StartM = FVector::ZeroVector;
	float StartYawDeg = 0.f;
	TArray<FResearchSpawnPoint> Spawns;
	int32 BoxCount = 0;
	// Phase 5 procedural variations (optional "appearance" object of the layout file).
	float Lighting = 1.f;          // global brightness factor (sun intensity)
	float ChestValue = 1.f;        // chest colour brightness factor
	float ChestHueShiftDeg = 0.f;  // chest colour hue shift
};

UCLASS()
class CHESTHUNTERRESEARCH_API UResearchArenaBuilder : public UObject
{
	GENERATED_BODY()

public:
	/** Parse the layout file (no world changes). */
	static bool LoadLayout(const FString& MapId, FResearchArenaLayout& Out, TSharedPtr<FJsonObject>& OutJson,
	                       FString& OutError);
	/** Spawn the static geometry into World (idempotent: previously built geometry is reused). */
	static bool BuildStatic(UWorld* World, const TSharedPtr<FJsonObject>& Json, TArray<AActor*>& InOutSpawned,
	                        FString& OutError);
	static FString LayoutPath(const FString& MapId);
	/** Map ids with a layout file in Content/ResearchData (research_arena_3d, proc_*). */
	static TArray<FString> AvailableMaps();
};
