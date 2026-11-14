// Unreal automation tests for the bridge protocol and the shared deterministic randomness.
// Run: UnrealEditor-Cmd ChestHunterResearch.uproject -ExecCmds="Automation RunTests ChestHunter;Quit"
//      -unattended -nullrhi -nosplash -log
// The expected values below were produced by the Python implementation
// (chest_hunter.simulation3d.environment3d.splitmix64 / unit_roll and protocol.encode_message),
// so these tests check that both sides agree bit for bit.

#include "Misc/AutomationTest.h"
#include "ResearchProtocol.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FResearchProtocolFramingTest, "ChestHunter.Protocol.Framing",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EngineFilter)

bool FResearchProtocolFramingTest::RunTest(const FString& Parameters)
{
	using namespace ResearchProtocol;
	TSharedRef<FJsonObject> Header = MakeShared<FJsonObject>();
	Header->SetStringField(TEXT("command"), TEXT("PING"));
	TArray<uint8> Bytes;
	FError Error;
	TestTrue(TEXT("encode"), EncodeMessage(Header, TArray<uint8>(), Bytes, Error));
	FPrefix Prefix;
	TestTrue(TEXT("prefix parses"), ParsePrefix(Bytes.GetData(), PrefixSize, Prefix, Error));
	TestEqual(TEXT("version"), int32(Prefix.Version), int32(Version));
	TestEqual(TEXT("payload"), int32(Prefix.PayloadLength), 0);
	TSharedPtr<FJsonObject> Parsed;
	TestTrue(TEXT("header parses"), ParseHeader(Bytes.GetData() + PrefixSize, Prefix.HeaderLength, Parsed, Error));
	TestEqual(TEXT("command round trip"), Parsed->GetStringField(TEXT("command")), FString(TEXT("PING")));

	// Python: encode_message({...}) prefix starts 43485242 0001 0000.
	TestEqual(TEXT("magic C"), int32(Bytes[0]), 0x43);
	TestEqual(TEXT("version hi"), int32(Bytes[4]), 0x00);
	TestEqual(TEXT("version lo"), int32(Bytes[5]), 0x01);

	TArray<uint8> Bad = Bytes;
	Bad[0] = 'X';
	TestFalse(TEXT("bad magic"), ParsePrefix(Bad.GetData(), PrefixSize, Prefix, Error));
	TestEqual(TEXT("bad magic code"), Error.Code, FString(Errors::BadMagic));
	Bad = Bytes;
	Bad[5] = 2;
	TestFalse(TEXT("version mismatch"), ParsePrefix(Bad.GetData(), PrefixSize, Prefix, Error));
	TestEqual(TEXT("version code"), Error.Code, FString(Errors::VersionMismatch));
	Bad = Bytes;
	Bad[12] = 0x7F;                                           // payload length far above 8 MiB
	TestFalse(TEXT("too large"), ParsePrefix(Bad.GetData(), PrefixSize, Prefix, Error));
	TestEqual(TEXT("too large code"), Error.Code, FString(Errors::TooLarge));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FResearchProtocolActionTest, "ChestHunter.Protocol.ActionValidation",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EngineFilter)

bool FResearchProtocolActionTest::RunTest(const FString& Parameters)
{
	using namespace ResearchProtocol;
	auto Make = [](double Forward) {
		TSharedPtr<FJsonObject> A = MakeShared<FJsonObject>();
		A->SetNumberField(TEXT("move_forward"), Forward);
		A->SetNumberField(TEXT("move_right"), 0.2);
		A->SetNumberField(TEXT("look_yaw"), 0.1);
		A->SetNumberField(TEXT("look_pitch"), -0.05);
		A->SetBoolField(TEXT("jump"), false);
		A->SetBoolField(TEXT("interact"), false);
		return A;
	};
	FAction Action;
	FError Error;
	TestTrue(TEXT("valid action"), ValidateAction(Make(0.7), Action, Error));
	TestEqual(TEXT("forward"), Action.MoveForward, 0.7f);
	TestFalse(TEXT("out of range"), ValidateAction(Make(2.0), Action, Error));
	TestEqual(TEXT("range code"), Error.Code, FString(Errors::InvalidAction));
	TSharedPtr<FJsonObject> Unknown = Make(0.0);
	Unknown->SetNumberField(TEXT("teleport"), 1.0);
	TestFalse(TEXT("unknown key"), ValidateAction(Unknown, Action, Error));
	TSharedPtr<FJsonObject> Missing = Make(0.0);
	Missing->RemoveField(TEXT("interact"));
	TestFalse(TEXT("missing key"), ValidateAction(Missing, Action, Error));
	TSharedPtr<FJsonObject> WrongType = Make(0.0);
	WrongType->SetNumberField(TEXT("jump"), 1.0);
	TestFalse(TEXT("bool must be bool"), ValidateAction(WrongType, Action, Error));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FResearchSpawnRollTest, "ChestHunter.Episode.DeterministicSpawns",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EngineFilter)

bool FResearchSpawnRollTest::RunTest(const FString& Parameters)
{
	using namespace ResearchProtocol;
	TestEqual(TEXT("splitmix64(0)"), SplitMix64(0), 0xe220a8397b1dcdafull);
	TestEqual(TEXT("splitmix64(42)"), SplitMix64(42), 0xbdd732262feb6e95ull);
	TestTrue(TEXT("unit_roll(42,0,0,0)"), FMath::IsNearlyEqual(UnitRoll(42, 0, 0, 0), 0.6595024198910419, 1e-15));
	TestTrue(TEXT("unit_roll(42,0,5,0)"), FMath::IsNearlyEqual(UnitRoll(42, 0, 5, 0), 0.8423371553602427, 1e-15));
	TestTrue(TEXT("unit_roll(7,3,12,1)"), FMath::IsNearlyEqual(UnitRoll(7, 3, 12, 1), 0.8942040471837918, 1e-15));
	TestTrue(TEXT("unit_roll(42,0,0,999)"), FMath::IsNearlyEqual(UnitRoll(42, 0, 0, 999), 0.4840224294029768, 1e-15));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
