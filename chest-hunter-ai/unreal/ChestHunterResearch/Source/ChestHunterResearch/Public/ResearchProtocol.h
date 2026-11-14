// Chest Hunter research bridge protocol (mirror of src/chest_hunter/unreal_bridge/protocol.py).
//
// Message = 16-byte prefix | UTF-8 JSON header | raw payload
//   prefix: "CHRB" | uint16 version | uint16 flags | uint32 header length | uint32 payload length
//   (all big-endian). Header 2..64 KiB, payload <= 8 MiB.
//
// Units on the wire are metres and degrees with Unreal's axes (X north/forward, Y east/right,
// Z up); the game code converts to centimetres.

#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"

namespace ResearchProtocol
{
	inline constexpr uint8 Magic[4] = { 'C', 'H', 'R', 'B' };
	inline constexpr uint16 Version = 1;
	inline constexpr int32 PrefixSize = 16;
	inline constexpr uint32 MaxHeaderBytes = 64 * 1024;
	inline constexpr uint32 MaxPayloadBytes = 8 * 1024 * 1024;
	inline constexpr int32 MaxTicksPerStep = 30;
	inline constexpr int32 MaxRays = 32;
	inline const TCHAR* const ProtocolName = TEXT("chest-hunter-bridge");

	/** Error codes shared with the Python side (protocol.ERROR_CODES). */
	namespace Errors
	{
		inline const TCHAR* const BadMagic = TEXT("bad_magic");
		inline const TCHAR* const VersionMismatch = TEXT("version_mismatch");
		inline const TCHAR* const TooLarge = TEXT("too_large");
		inline const TCHAR* const BadJson = TEXT("bad_json");
		inline const TCHAR* const BadHeader = TEXT("bad_header");
		inline const TCHAR* const UnknownCommand = TEXT("unknown_command");
		inline const TCHAR* const InvalidArgument = TEXT("invalid_argument");
		inline const TCHAR* const InvalidAction = TEXT("invalid_action");
		inline const TCHAR* const NoEpisode = TEXT("no_episode");
		inline const TCHAR* const StaleEpisode = TEXT("stale_episode");
		inline const TCHAR* const StaleStep = TEXT("stale_step");
		inline const TCHAR* const OutOfOrder = TEXT("out_of_order");
		inline const TCHAR* const EpisodeEnded = TEXT("episode_ended");
		inline const TCHAR* const NavigationDisabled = TEXT("navigation_disabled");
		inline const TCHAR* const Forbidden = TEXT("forbidden");
		inline const TCHAR* const Busy = TEXT("busy");
		inline const TCHAR* const Internal = TEXT("internal");
	}

	/** Errors that make the byte stream untrustworthy: the connection is closed. */
	bool IsFatal(const FString& Code);

	struct FPrefix
	{
		uint16 Version = 0;
		uint32 HeaderLength = 0;
		uint32 PayloadLength = 0;
	};

	struct FError
	{
		FString Code;
		FString Message;
	};

	/** Validate the 16-byte prefix. */
	bool ParsePrefix(const uint8* Data, int32 Length, FPrefix& Out, FError& OutError);

	/** Parse a UTF-8 JSON header into an object (non-finite numbers are rejected). */
	bool ParseHeader(const uint8* Data, int32 Length, TSharedPtr<FJsonObject>& Out, FError& OutError);

	/** Encode a full message. Fails (returns false) if a size bound would be exceeded. */
	bool EncodeMessage(const TSharedRef<FJsonObject>& Header, const TArray<uint8>& Payload,
	                   TArray<uint8>& OutBytes, FError& OutError);

	/** One validated action (analogue axes in [-1, 1]; look values are fractions of max rate). */
	struct FAction
	{
		float MoveForward = 0.f;
		float MoveRight = 0.f;
		float LookYaw = 0.f;
		float LookPitch = 0.f;
		bool bJump = false;
		bool bInteract = false;
		bool bSprint = false;
		bool bCrouch = false;
	};

	/** Same rules as protocol.validate_action: required keys, no unknown keys, ranges, types. */
	bool ValidateAction(const TSharedPtr<FJsonObject>& Object, FAction& Out, FError& OutError);

	/** Envelope check: command name, request_id, args object (argument checks are per command). */
	bool ValidateEnvelope(const TSharedPtr<FJsonObject>& Header, FString& OutCommand,
	                      int64& OutRequestId, TSharedPtr<FJsonObject>& OutArgs, FError& OutError);

	/** Argument helpers used by the command handlers (all reject wrong types and ranges). */
	bool GetNumber(const TSharedPtr<FJsonObject>& Args, const FString& Key, double Lo, double Hi,
	               bool bRequired, double Default, double& Out, FError& OutError);
	bool GetInteger(const TSharedPtr<FJsonObject>& Args, const FString& Key, int64 Lo, int64 Hi,
	                bool bRequired, int64 Default, int64& Out, FError& OutError);
	bool GetString(const TSharedPtr<FJsonObject>& Args, const FString& Key, bool bRequired,
	               FString& Out, FError& OutError);
	bool GetBool(const TSharedPtr<FJsonObject>& Args, const FString& Key, bool Default, bool& Out,
	             FError& OutError);
	bool GetVector(const TSharedPtr<FJsonObject>& Args, const FString& Key, FVector& OutMetres,
	               FError& OutError);
	bool CheckKeys(const TSharedPtr<FJsonObject>& Args, const TArray<FString>& Allowed,
	               FError& OutError);

	/** Response builders. */
	TSharedRef<FJsonObject> OkResponse(const FString& Command, int64 RequestId);
	TSharedRef<FJsonObject> ErrorResponse(const FString& Command, int64 RequestId, const FError& Error);

	/** Deterministic randomness shared with the Python environment (environment3d.unit_roll). */
	uint64 SplitMix64(uint64 Value);
	double UnitRoll(int32 Seed, int32 Episode, int32 Index, int32 Salt);
}
