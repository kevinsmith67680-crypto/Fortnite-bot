// Chest Hunter research environment: primary game module.
// A self-contained research arena (UE5 starter content and engine basic shapes only,
// no third-party game assets). The agent controls the character exclusively through the
// loopback bridge implemented in ResearchBridgeServer / ResearchBridgeSubsystem.

#pragma once

#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LogChestHunter, Log, All);

/** Unreal works in centimetres; the bridge protocol in metres. */
namespace ResearchUnits
{
	inline constexpr double CmPerMetre = 100.0;
	inline FVector ToCm(const FVector& Metres) { return Metres * CmPerMetre; }
	inline FVector ToMetres(const FVector& Cm) { return Cm / CmPerMetre; }
}
