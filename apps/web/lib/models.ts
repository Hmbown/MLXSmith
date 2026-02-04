import type { JSONValue, ModelInfo } from "@/types";
import { formatBytes } from "@/lib/utils";

export interface LocalModel extends ModelInfo {
  displayName: string;
  fullName: string;
  sizeDisplay: string;
  quantizationDisplay: string;
  baseModel?: string;
  architecture?: string;
  isAdapter: boolean;
  cliArgument: string;
}

function stringValue(value: JSONValue | undefined): string | undefined {
  if (typeof value === "string") return value;
  if (typeof value === "number") return value.toString();
  if (typeof value === "boolean") return value ? "true" : "false";
  return undefined;
}

function arrayStringValue(value: JSONValue | undefined): string | undefined {
  if (!Array.isArray(value)) return undefined;
  for (const entry of value) {
    const s = stringValue(entry);
    if (s) return s;
  }
  return undefined;
}

export function toLocalModel(info: ModelInfo): LocalModel {
  const cleanedId = info.id.replace(" (HF)", "");
  const displayName = cleanedId.split("/").pop() || cleanedId;
  const metadata = info.metadata || {};

  const quantization =
    stringValue(metadata["quantization"] as JSONValue) ||
    (() => {
      const lower = cleanedId.toLowerCase();
      if (lower.includes("4bit") || lower.includes("4-bit")) return "4-bit";
      if (lower.includes("8bit") || lower.includes("8-bit")) return "8-bit";
      return "none";
    })();

  const baseModel =
    stringValue(metadata["base_model"] as JSONValue) ||
    stringValue(metadata["base_model_name_or_path"] as JSONValue) ||
    stringValue(metadata["base_model_id"] as JSONValue);

  const architecture =
    stringValue(metadata["model_type"] as JSONValue) ||
    stringValue(metadata["architecture"] as JSONValue) ||
    arrayStringValue(metadata["architectures"] as JSONValue);

  const isAdapter = Boolean(info.has_adapter || info.adapter_path || info.path.includes("adapter"));
  const cliArgument = info.has_adapter && info.adapter_path
    ? info.adapter_path
    : info.format === "hf"
      ? cleanedId
      : info.path;

  return {
    ...info,
    displayName,
    fullName: cleanedId,
    sizeDisplay: info.size_bytes ? formatBytes(info.size_bytes) : "—",
    quantizationDisplay: quantization,
    baseModel: baseModel || undefined,
    architecture: architecture || undefined,
    isAdapter,
    cliArgument,
  };
}
