using UnrealBuildTool;

public class ChestHunterResearch : ModuleRules
{
	public ChestHunterResearch(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		PublicDependencyModuleNames.AddRange(new string[] {
			"Core", "CoreUObject", "Engine", "InputCore", "Json", "JsonUtilities",
			"Sockets", "Networking", "NavigationSystem", "AIModule", "RenderCore"
		});
	}
}
