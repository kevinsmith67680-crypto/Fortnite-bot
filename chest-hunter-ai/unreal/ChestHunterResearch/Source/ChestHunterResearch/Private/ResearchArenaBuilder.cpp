#include "ResearchArenaBuilder.h"

#include "ChestHunterResearch.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/World.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
	bool ReadVec(const TSharedPtr<FJsonObject>& Obj, const FString& Key, FVector& Out)
	{
		const TArray<TSharedPtr<FJsonValue>>* A = nullptr;
		if (!Obj->TryGetArrayField(Key, A) || A->Num() != 3)
		{
			return false;
		}
		Out = FVector((*A)[0]->AsNumber(), (*A)[1]->AsNumber(), (*A)[2]->AsNumber());
		return true;
	}

	bool IsSafeMapId(const FString& MapId)
	{
		// Only plain identifiers: the id selects a file under Content/ResearchData.
		if (MapId.IsEmpty() || MapId.Len() > 64) { return false; }
		for (const TCHAR C : MapId)
		{
			if (!(FChar::IsAlnum(C) || C == TEXT('_') || C == TEXT('-'))) { return false; }
		}
		return true;
	}
}

FString UResearchArenaBuilder::LayoutPath(const FString& MapId)
{
	return FPaths::Combine(FPaths::ProjectContentDir(), TEXT("ResearchData"), MapId + TEXT(".json"));
}

bool UResearchArenaBuilder::LoadLayout(const FString& MapId, FResearchArenaLayout& Out, TSharedPtr<FJsonObject>& OutJson,
                                       FString& OutError)
{
	if (!IsSafeMapId(MapId))
	{
		OutError = TEXT("map_id must be a plain identifier");
		return false;
	}
	FString Text;
	const FString Path = LayoutPath(MapId);
	if (!FFileHelper::LoadFileToString(Text, *Path))
	{
		OutError = FString::Printf(TEXT("unknown map '%s' (no layout file %s)"), *MapId, *Path);
		return false;
	}
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
	if (!FJsonSerializer::Deserialize(Reader, OutJson) || !OutJson.IsValid())
	{
		OutError = TEXT("layout file is not valid JSON");
		return false;
	}
	if (OutJson->GetStringField(TEXT("format")) != TEXT("chest-hunter-arena/1"))
	{
		OutError = TEXT("unsupported layout format");
		return false;
	}
	Out.MapId = OutJson->GetStringField(TEXT("map_id"));
	const TArray<TSharedPtr<FJsonValue>>& Bounds = OutJson->GetArrayField(TEXT("bounds"));
	if (Bounds.Num() == 4)
	{
		Out.BoundsM = FBox2D(FVector2D(Bounds[0]->AsNumber(), Bounds[1]->AsNumber()),
		                     FVector2D(Bounds[2]->AsNumber(), Bounds[3]->AsNumber()));
	}
	const TSharedPtr<FJsonObject> Start = OutJson->GetObjectField(TEXT("start"));
	ReadVec(Start, TEXT("position"), Out.StartM);
	Out.StartYawDeg = float(Start->GetNumberField(TEXT("yaw_deg")));
	Out.Spawns.Reset();
	int32 Index = 0;
	for (const TSharedPtr<FJsonValue>& V : OutJson->GetArrayField(TEXT("spawn_points")))
	{
		const TSharedPtr<FJsonObject> S = V->AsObject();
		FResearchSpawnPoint P;
		P.Index = Index++;
		P.Region = S->GetStringField(TEXT("region"));
		P.Name = S->GetStringField(TEXT("name"));
		P.LocationName = S->GetStringField(TEXT("location_name"));
		P.PositionM = FVector(S->GetNumberField(TEXT("x")), S->GetNumberField(TEXT("y")), S->GetNumberField(TEXT("z")));
		P.YawDeg = float(S->GetNumberField(TEXT("yaw_deg")));
		P.Probability = float(S->GetNumberField(TEXT("probability")));
		S->TryGetBoolField(TEXT("reachable"), P.bReachable);
		Out.Spawns.Add(P);
	}
	Out.BoxCount = OutJson->GetArrayField(TEXT("boxes")).Num();
	const TSharedPtr<FJsonObject>* Appearance = nullptr;
	if (OutJson->TryGetObjectField(TEXT("appearance"), Appearance))
	{
		double Value = 1.0;
		if ((*Appearance)->TryGetNumberField(TEXT("lighting"), Value)) { Out.Lighting = float(Value); }
		if ((*Appearance)->TryGetNumberField(TEXT("chest_value"), Value)) { Out.ChestValue = float(Value); }
		if ((*Appearance)->TryGetNumberField(TEXT("chest_hue_shift"), Value)) { Out.ChestHueShiftDeg = float(Value); }
	}
	return true;
}

TArray<FString> UResearchArenaBuilder::AvailableMaps()
{
	TArray<FString> Files;
	IFileManager::Get().FindFiles(Files, *FPaths::Combine(FPaths::ProjectContentDir(), TEXT("ResearchData"), TEXT("*.json")), true, false);
	TArray<FString> Ids;
	for (const FString& File : Files)
	{
		const FString Id = FPaths::GetBaseFilename(File);
		if (IsSafeMapId(Id)) { Ids.Add(Id); }
	}
	Ids.Sort();
	return Ids;
}

bool UResearchArenaBuilder::BuildStatic(UWorld* World, const TSharedPtr<FJsonObject>& Json, TArray<AActor*>& InOutSpawned,
                                        FString& OutError)
{
	if (InOutSpawned.Num() > 0)
	{
		return true;                                     // static geometry never changes between episodes
	}
	UStaticMesh* Cube = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	UMaterialInterface* Base = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
	if (Cube == nullptr)
	{
		OutError = TEXT("engine basic shapes are missing");
		return false;
	}
	TMap<FString, FLinearColor> Colours;
	const TSharedPtr<FJsonObject> Palette = Json->GetObjectField(TEXT("colours_bgr"));
	for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Palette->Values)
	{
		const TArray<TSharedPtr<FJsonValue>>& Bgr = Pair.Value->AsArray();
		Colours.Add(Pair.Key, FLinearColor(FColor(uint8(Bgr[2]->AsNumber()), uint8(Bgr[1]->AsNumber()), uint8(Bgr[0]->AsNumber()))));
	}
	TMap<FString, UMaterialInstanceDynamic*> Materials;
	for (const TSharedPtr<FJsonValue>& V : Json->GetArrayField(TEXT("boxes")))
	{
		const TSharedPtr<FJsonObject> B = V->AsObject();
		FVector Min, Max;
		if (!ReadVec(B, TEXT("min"), Min) || !ReadVec(B, TEXT("max"), Max))
		{
			OutError = TEXT("malformed box in layout");
			return false;
		}
		const FVector CentreCm = ResearchUnits::ToCm((Min + Max) * 0.5);
		const FVector SizeM = Max - Min;
		FActorSpawnParameters Params;
		Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		AStaticMeshActor* Actor = World->SpawnActor<AStaticMeshActor>(CentreCm, FRotator::ZeroRotator, Params);
		if (Actor == nullptr)
		{
			continue;
		}
		UStaticMeshComponent* Mesh = Actor->GetStaticMeshComponent();
		Mesh->SetMobility(EComponentMobility::Movable);
		Mesh->SetStaticMesh(Cube);
		Actor->SetActorScale3D(SizeM);                    // engine cube = 1 m
		const bool bBlocks = B->GetBoolField(TEXT("blocks"));
		Mesh->SetCollisionProfileName(bBlocks ? TEXT("BlockAll") : TEXT("NoCollision"));
		Mesh->SetCanEverAffectNavigation(bBlocks);
		const FString Colour = B->GetStringField(TEXT("colour"));
		if (Base != nullptr)
		{
			UMaterialInstanceDynamic*& Mid = Materials.FindOrAdd(Colour);
			if (Mid == nullptr)
			{
				Mid = UMaterialInstanceDynamic::Create(Base, World);
				Mid->SetVectorParameterValue(TEXT("Color"), Colours.FindRef(Colour));
			}
			Mesh->SetMaterial(0, Mid);
		}
		Actor->Tags.Add(FName(*FString::Printf(TEXT("ResearchArena:%s"), *B->GetStringField(TEXT("kind")))));
#if WITH_EDITOR
		Actor->SetActorLabel(B->GetStringField(TEXT("name")).IsEmpty() ? Colour : B->GetStringField(TEXT("name")));
#endif
		InOutSpawned.Add(Actor);
	}
	UE_LOG(LogChestHunter, Log, TEXT("Research arena built: %d boxes"), InOutSpawned.Num());
	return true;
}
