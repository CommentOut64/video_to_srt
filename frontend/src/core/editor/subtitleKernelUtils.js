import { toRaw } from "vue";

export function secondsToMs(value) {
  const normalized = Number(value);
  if (!Number.isFinite(normalized)) return 0;
  return Math.round(normalized * 1000);
}

export function msToSeconds(value) {
  const normalized = Number(value);
  if (!Number.isFinite(normalized)) return 0;
  return normalized / 1000;
}

function normalizeString(value, fallback = "") {
  if (value === undefined || value === null) return fallback;
  return String(value);
}

function normalizeNullable(value) {
  return value === undefined ? null : value;
}

export function buildHotSubtitleRecord(subtitle = {}) {
  const rawSubtitle = toRaw(subtitle) || {};
  const startMs = rawSubtitle.startMs ?? secondsToMs(rawSubtitle.start ?? 0);
  const endMs = rawSubtitle.endMs ?? secondsToMs(rawSubtitle.end ?? 0);

  return {
    localId: normalizeString(rawSubtitle.id ?? rawSubtitle.localId),
    text: normalizeString(rawSubtitle.text),
    startMs,
    endMs,
    flags: {
      isDirty: Boolean(rawSubtitle.isDirty),
      isModified: Boolean(rawSubtitle.isModified),
      isDraft: Boolean(rawSubtitle.isDraft),
      isFinalized: rawSubtitle.isFinalized ?? !Boolean(rawSubtitle.isDraft),
      warningType: rawSubtitle.warning_type || "none",
    },
    originalText: normalizeNullable(rawSubtitle.originalText),
    chunkId: normalizeNullable(rawSubtitle.chunk_id),
    source: rawSubtitle.source || "manual",
    revision: Number.isFinite(Number(rawSubtitle.revision)) ? Number(rawSubtitle.revision) : 0,
  };
}

export function buildColdSubtitleRecord(subtitle = {}) {
  const rawSubtitle = toRaw(subtitle) || {};
  return {
    localId: normalizeString(rawSubtitle.id ?? rawSubtitle.localId),
    segmentId: normalizeNullable(rawSubtitle.segment_id),
    sentenceIndex: Number.isFinite(Number(rawSubtitle.sentenceIndex)) ? Number(rawSubtitle.sentenceIndex) : null,
    words: Array.isArray(rawSubtitle.words) ? rawSubtitle.words : [],
    confidence: rawSubtitle.confidence ?? null,
    displayConfidence: rawSubtitle.display_confidence ?? null,
    confidenceSource: rawSubtitle.confidence_source ?? null,
    speakerId: rawSubtitle.speaker_id ?? null,
    speakerName: rawSubtitle.speaker_name ?? null,
    speakerColor: rawSubtitle.speaker_color ?? null,
    speakerLocked: Boolean(rawSubtitle.speaker_locked),
    speakerConfidence: rawSubtitle.speaker_confidence ?? null,
    sourceMeta: rawSubtitle.source_meta ?? null,
  };
}

export function buildSubtitleProjection(hotRecord, coldRecord = null) {
  if (!hotRecord?.localId) return null;

  return {
    id: hotRecord.localId,
    localId: hotRecord.localId,
    text: hotRecord.text,
    start: msToSeconds(hotRecord.startMs),
    end: msToSeconds(hotRecord.endMs),
    startMs: hotRecord.startMs,
    endMs: hotRecord.endMs,
    isDirty: Boolean(hotRecord.flags?.isDirty),
    isModified: Boolean(hotRecord.flags?.isModified),
    isDraft: Boolean(hotRecord.flags?.isDraft),
    isFinalized: hotRecord.flags?.isFinalized ?? !Boolean(hotRecord.flags?.isDraft),
    warning_type: hotRecord.flags?.warningType || "none",
    originalText: hotRecord.originalText ?? null,
    chunk_id: hotRecord.chunkId ?? null,
    source: hotRecord.source || "manual",
    revision: hotRecord.revision ?? 0,
    segment_id: coldRecord?.segmentId ?? null,
    sentenceIndex: coldRecord?.sentenceIndex ?? null,
    words: Array.isArray(coldRecord?.words) ? coldRecord.words : [],
    confidence: coldRecord?.confidence ?? null,
    display_confidence: coldRecord?.displayConfidence ?? null,
    confidence_source: coldRecord?.confidenceSource ?? null,
    speaker_id: coldRecord?.speakerId ?? null,
    speaker_name: coldRecord?.speakerName ?? null,
    speaker_color: coldRecord?.speakerColor ?? null,
    speaker_locked: Boolean(coldRecord?.speakerLocked),
    speaker_confidence: coldRecord?.speakerConfidence ?? null,
    source_meta: coldRecord?.sourceMeta ?? null,
  };
}
