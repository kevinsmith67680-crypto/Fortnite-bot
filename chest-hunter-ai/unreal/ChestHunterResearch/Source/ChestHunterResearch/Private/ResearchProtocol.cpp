#include "ResearchProtocol.h"

#include "Policies/CondensedJsonPrintPolicy.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace ResearchProtocol
{
	namespace
	{
		uint32 ReadU32(const uint8* P) { return (uint32(P[0]) << 24) | (uint32(P[1]) << 16) | (uint32(P[2]) << 8) | uint32(P[3]); }
		uint16 ReadU16(const uint8* P) { return uint16((uint32(P[0]) << 8) | uint32(P[1])); }
		void WriteU32(TArray<uint8>& Out, uint32 V)
		{
			Out.Add(uint8(V >> 24)); Out.Add(uint8(V >> 16)); Out.Add(uint8(V >> 8)); Out.Add(uint8(V));
		}
		void WriteU16(TArray<uint8>& Out, uint16 V) { Out.Add(uint8(V >> 8)); Out.Add(uint8(V)); }

		FError Err(const TCHAR* Code, const FString& Message) { return FError{ FString(Code), Message }; }

		const TArray<FString>& FloatActions()
		{
			static const TArray<FString> Keys = { TEXT("move_forward"), TEXT("move_right"), TEXT("look_yaw"), TEXT("look_pitch") };
			return Keys;
		}
		const TArray<FString>& BoolActions()
		{
			static const TArray<FString> Keys = { TEXT("jump"), TEXT("interact"), TEXT("sprint"), TEXT("crouch") };
			return Keys;
		}
		const TArray<FString>& Commands()
		{
			static const TArray<FString> Names = {
				TEXT("HELLO"), TEXT("RESET_EPISODE"), TEXT("GET_OBSERVATION"), TEXT("STEP"),
				TEXT("GET_EPISODE_STATUS"), TEXT("STOP_EPISODE"), TEXT("PING"), TEXT("NAV_QUERY"),
				TEXT("NAV_RAYCAST"), TEXT("GET_ORACLE_STATE"), TEXT("GET_EVALUATION") };
			return Names;
		}
	}

	bool IsFatal(const FString& Code)
	{
		return Code == Errors::BadMagic || Code == Errors::VersionMismatch || Code == Errors::TooLarge
			|| Code == Errors::BadJson || Code == Errors::BadHeader;
	}

	bool ParsePrefix(const uint8* Data, int32 Length, FPrefix& Out, FError& OutError)
	{
		if (Length != PrefixSize)
		{
			OutError = Err(Errors::BadHeader, TEXT("truncated message prefix"));
			return false;
		}
		if (FMemory::Memcmp(Data, Magic, 4) != 0)
		{
			OutError = Err(Errors::BadMagic, TEXT("expected magic CHRB"));
			return false;
		}
		Out.Version = ReadU16(Data + 4);
		Out.HeaderLength = ReadU32(Data + 8);
		Out.PayloadLength = ReadU32(Data + 12);
		if (Out.Version != Version)
		{
			OutError = Err(Errors::VersionMismatch, FString::Printf(TEXT("peer speaks protocol version %u, server %u"), Out.Version, Version));
			return false;
		}
		if (Out.HeaderLength < 2 || Out.HeaderLength > MaxHeaderBytes)
		{
			OutError = Err(Out.HeaderLength > MaxHeaderBytes ? Errors::TooLarge : Errors::BadHeader,
			               FString::Printf(TEXT("header length %u outside [2, %u]"), Out.HeaderLength, MaxHeaderBytes));
			return false;
		}
		if (Out.PayloadLength > MaxPayloadBytes)
		{
			OutError = Err(Errors::TooLarge, FString::Printf(TEXT("payload length %u exceeds %u"), Out.PayloadLength, MaxPayloadBytes));
			return false;
		}
		return true;
	}

	bool ParseHeader(const uint8* Data, int32 Length, TSharedPtr<FJsonObject>& Out, FError& OutError)
	{
		const FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(Data), Length);
		const FString Text(Converted.Length(), Converted.Get());
		// The Python side never emits NaN/Infinity (allow_nan=False); reject them here too.
		if (Text.Contains(TEXT("NaN")) || Text.Contains(TEXT("Infinity")))
		{
			OutError = Err(Errors::BadJson, TEXT("non-finite numbers are not allowed"));
			return false;
		}
		const TSharedRef<TJsonReader<TCHAR>> Reader = TJsonReaderFactory<TCHAR>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Out) || !Out.IsValid())
		{
			OutError = Err(Errors::BadJson, TEXT("header is not a valid JSON object"));
			return false;
		}
		return true;
	}

	bool EncodeMessage(const TSharedRef<FJsonObject>& Header, const TArray<uint8>& Payload,
	                   TArray<uint8>& OutBytes, FError& OutError)
	{
		FString Text;
		const TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
			TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Text);
		if (!FJsonSerializer::Serialize(Header, Writer))
		{
			OutError = Err(Errors::BadHeader, TEXT("header could not be serialised"));
			return false;
		}
		const FTCHARToUTF8 Utf8(*Text);
		if (uint32(Utf8.Length()) > MaxHeaderBytes || uint32(Payload.Num()) > MaxPayloadBytes)
		{
			OutError = Err(Errors::TooLarge, TEXT("message exceeds the protocol size bounds"));
			return false;
		}
		OutBytes.Reset(PrefixSize + Utf8.Length() + Payload.Num());
		OutBytes.Append(Magic, 4);
		WriteU16(OutBytes, Version);
		WriteU16(OutBytes, 0);
		WriteU32(OutBytes, uint32(Utf8.Length()));
		WriteU32(OutBytes, uint32(Payload.Num()));
		OutBytes.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
		OutBytes.Append(Payload);
		return true;
	}

	bool CheckKeys(const TSharedPtr<FJsonObject>& Args, const TArray<FString>& Allowed, FError& OutError)
	{
		for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Args->Values)
		{
			if (!Allowed.Contains(Pair.Key))
			{
				OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("unexpected argument '%s'"), *Pair.Key));
				return false;
			}
		}
		return true;
	}

	bool ValidateAction(const TSharedPtr<FJsonObject>& Object, FAction& Out, FError& OutError)
	{
		if (!Object.IsValid())
		{
			OutError = Err(Errors::InvalidAction, TEXT("action must be a JSON object"));
			return false;
		}
		TArray<FString> Allowed = FloatActions();
		Allowed.Append(BoolActions());
		for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Object->Values)
		{
			if (!Allowed.Contains(Pair.Key))
			{
				OutError = Err(Errors::InvalidAction, FString::Printf(TEXT("unknown action key '%s'"), *Pair.Key));
				return false;
			}
		}
		float* Floats[] = { &Out.MoveForward, &Out.MoveRight, &Out.LookYaw, &Out.LookPitch };
		for (int32 I = 0; I < FloatActions().Num(); ++I)
		{
			const TSharedPtr<FJsonValue> V = Object->TryGetField(FloatActions()[I]);
			if (!V.IsValid() || V->Type != EJson::Number)
			{
				OutError = Err(Errors::InvalidAction, FString::Printf(TEXT("%s must be a number"), *FloatActions()[I]));
				return false;
			}
			const double D = V->AsNumber();
			if (!FMath::IsFinite(D) || D < -1.0 || D > 1.0)
			{
				OutError = Err(Errors::InvalidAction, FString::Printf(TEXT("%s must be within [-1, 1]"), *FloatActions()[I]));
				return false;
			}
			*Floats[I] = float(D);
		}
		bool* Bools[] = { &Out.bJump, &Out.bInteract, &Out.bSprint, &Out.bCrouch };
		for (int32 I = 0; I < BoolActions().Num(); ++I)
		{
			const TSharedPtr<FJsonValue> V = Object->TryGetField(BoolActions()[I]);
			const bool bRequired = I < 2;                         // jump and interact are required
			if (!V.IsValid())
			{
				if (bRequired)
				{
					OutError = Err(Errors::InvalidAction, FString::Printf(TEXT("missing action key '%s'"), *BoolActions()[I]));
					return false;
				}
				*Bools[I] = false;
				continue;
			}
			if (V->Type != EJson::Boolean)
			{
				OutError = Err(Errors::InvalidAction, FString::Printf(TEXT("%s must be true or false"), *BoolActions()[I]));
				return false;
			}
			*Bools[I] = V->AsBool();
		}
		return true;
	}

	bool ValidateEnvelope(const TSharedPtr<FJsonObject>& Header, FString& OutCommand, int64& OutRequestId,
	                      TSharedPtr<FJsonObject>& OutArgs, FError& OutError)
	{
		if (!Header->TryGetStringField(TEXT("command"), OutCommand))
		{
			OutError = Err(Errors::BadHeader, TEXT("missing 'command'"));
			return false;
		}
		if (!Commands().Contains(OutCommand))
		{
			OutError = Err(Errors::UnknownCommand, FString::Printf(TEXT("unknown command '%s'"), *OutCommand));
			return false;
		}
		const TSharedPtr<FJsonValue> Rid = Header->TryGetField(TEXT("request_id"));
		if (!Rid.IsValid() || Rid->Type != EJson::Number || Rid->AsNumber() < 1.0 || FMath::Frac(Rid->AsNumber()) != 0.0)
		{
			OutError = Err(Errors::InvalidArgument, TEXT("request_id must be a positive integer"));
			return false;
		}
		OutRequestId = int64(Rid->AsNumber());
		const TSharedPtr<FJsonObject>* Args = nullptr;
		if (Header->HasField(TEXT("args")))
		{
			if (!Header->TryGetObjectField(TEXT("args"), Args))
			{
				OutError = Err(Errors::InvalidArgument, TEXT("args must be an object"));
				return false;
			}
			OutArgs = *Args;
		}
		else
		{
			OutArgs = MakeShared<FJsonObject>();
		}
		return true;
	}

	bool GetNumber(const TSharedPtr<FJsonObject>& Args, const FString& Key, double Lo, double Hi, bool bRequired,
	               double Default, double& Out, FError& OutError)
	{
		const TSharedPtr<FJsonValue> V = Args->TryGetField(Key);
		if (!V.IsValid())
		{
			if (bRequired)
			{
				OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("'%s' is required"), *Key));
				return false;
			}
			Out = Default;
			return true;
		}
		if (V->Type != EJson::Number || !FMath::IsFinite(V->AsNumber()) || V->AsNumber() < Lo || V->AsNumber() > Hi)
		{
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s must be a number within [%g, %g]"), *Key, Lo, Hi));
			return false;
		}
		Out = V->AsNumber();
		return true;
	}

	bool GetInteger(const TSharedPtr<FJsonObject>& Args, const FString& Key, int64 Lo, int64 Hi, bool bRequired,
	                int64 Default, int64& Out, FError& OutError)
	{
		double D = 0.0;
		if (!GetNumber(Args, Key, double(Lo), double(Hi), bRequired, double(Default), D, OutError))
		{
			return false;
		}
		if (FMath::Frac(D) != 0.0)
		{
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s must be an integer"), *Key));
			return false;
		}
		Out = int64(D);
		return true;
	}

	bool GetString(const TSharedPtr<FJsonObject>& Args, const FString& Key, bool bRequired, FString& Out, FError& OutError)
	{
		if (!Args->TryGetStringField(Key, Out))
		{
			if (!bRequired && !Args->HasField(Key))
			{
				return true;
			}
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s must be a non-empty string"), *Key));
			return false;
		}
		if (Out.TrimStartAndEnd().IsEmpty() || Out.Len() > 256)
		{
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s must be a non-empty string"), *Key));
			return false;
		}
		return true;
	}

	bool GetBool(const TSharedPtr<FJsonObject>& Args, const FString& Key, bool Default, bool& Out, FError& OutError)
	{
		const TSharedPtr<FJsonValue> V = Args->TryGetField(Key);
		if (!V.IsValid())
		{
			Out = Default;
			return true;
		}
		if (V->Type != EJson::Boolean)
		{
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s must be true or false"), *Key));
			return false;
		}
		Out = V->AsBool();
		return true;
	}

	bool GetVector(const TSharedPtr<FJsonObject>& Args, const FString& Key, FVector& OutMetres, FError& OutError)
	{
		const TArray<TSharedPtr<FJsonValue>>* Items = nullptr;
		if (!Args->TryGetArrayField(Key, Items) || Items->Num() != 3)
		{
			OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s must be [x, y, z]"), *Key));
			return false;
		}
		double C[3];
		for (int32 I = 0; I < 3; ++I)
		{
			if ((*Items)[I]->Type != EJson::Number || !FMath::IsFinite((*Items)[I]->AsNumber())
				|| FMath::Abs((*Items)[I]->AsNumber()) > 1e5)
			{
				OutError = Err(Errors::InvalidArgument, FString::Printf(TEXT("%s[%d] must be a finite number"), *Key, I));
				return false;
			}
			C[I] = (*Items)[I]->AsNumber();
		}
		OutMetres = FVector(C[0], C[1], C[2]);
		return true;
	}

	TSharedRef<FJsonObject> OkResponse(const FString& Command, int64 RequestId)
	{
		TSharedRef<FJsonObject> R = MakeShared<FJsonObject>();
		R->SetStringField(TEXT("type"), TEXT("response"));
		R->SetStringField(TEXT("command"), Command);
		R->SetNumberField(TEXT("request_id"), double(RequestId));
		R->SetStringField(TEXT("status"), TEXT("ok"));
		return R;
	}

	TSharedRef<FJsonObject> ErrorResponse(const FString& Command, int64 RequestId, const FError& Error)
	{
		TSharedRef<FJsonObject> R = MakeShared<FJsonObject>();
		R->SetStringField(TEXT("type"), TEXT("response"));
		if (Command.IsEmpty()) { R->SetField(TEXT("command"), MakeShared<FJsonValueNull>()); }
		else { R->SetStringField(TEXT("command"), Command); }
		if (RequestId > 0) { R->SetNumberField(TEXT("request_id"), double(RequestId)); }
		else { R->SetField(TEXT("request_id"), MakeShared<FJsonValueNull>()); }
		R->SetStringField(TEXT("status"), TEXT("error"));
		TSharedRef<FJsonObject> E = MakeShared<FJsonObject>();
		E->SetStringField(TEXT("code"), Error.Code);
		E->SetStringField(TEXT("message"), Error.Message);
		R->SetObjectField(TEXT("error"), E);
		return R;
	}

	uint64 SplitMix64(uint64 Value)
	{
		uint64 Z = Value + 0x9E3779B97F4A7C15ull;
		Z = (Z ^ (Z >> 30)) * 0xBF58476D1CE4E5B9ull;
		Z = (Z ^ (Z >> 27)) * 0x94D049BB133111EBull;
		return Z ^ (Z >> 31);
	}

	double UnitRoll(int32 Seed, int32 Episode, int32 Index, int32 Salt)
	{
		uint64 H = SplitMix64((uint64(uint32(Seed)) << 32) | uint64(uint32(Episode)));
		H = SplitMix64(H ^ ((uint64(uint32(Index) & 0xFFFF) << 16) | uint64(uint32(Salt) & 0xFFFF)));
		return double(H >> 11) / double(1ull << 53);
	}
}
