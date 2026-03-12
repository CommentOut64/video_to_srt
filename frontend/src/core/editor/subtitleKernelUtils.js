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

function hasFiniteNumber(value) {
  return Number.isFinite(Number(value));
}

export function buildHotSubtitleRecord(subtitle = {}) {
  const rawSubtitle = toRaw(subtitle) || {};
  const startMs = hasFiniteNumber(rawSubtitle.start)
    ? secondsToMs(rawSubtitle.start)
    : (hasFiniteNumber(rawSubtitle.startMs) ? Number(rawSubtitle.startMs) : 0);
  const endMs = hasFiniteNumber(rawSubtitle.end)
    ? secondsToMs(rawSubtitle.end)
    : (hasFiniteNumber(rawSubtitle.endMs) ? Number(rawSubtitle.endMs) : startMs);

  return {
    localId: normalizeString(rawSubtitle.id ?? rawSubtitle.localId),
    text: normalizeString(rawSubtitle.text),
    startMs,
    endMs,
    flags: {
      isDirty: Boolean(rawSubtitle.isDirty ?? rawSubtitle.is_dirty),
      isModified: Boolean(rawSubtitle.isModified ?? rawSubtitle.is_modified),
      isDraft: Boolean(rawSubtitle.isDraft ?? rawSubtitle.is_draft),
      isFinalized: rawSubtitle.isFinalized ?? rawSubtitle.is_finalized ?? !Boolean(rawSubtitle.isDraft ?? rawSubtitle.is_draft),
      warningType: rawSubtitle.warning_type || "none",
    },
    originalText: normalizeNullable(rawSubtitle.originalText ?? rawSubtitle.original_text),
    chunkId: normalizeNullable(rawSubtitle.chunk_id ?? rawSubtitle.chunkId),
    source: rawSubtitle.source || "manual",
    revision: Number.isFinite(Number(rawSubtitle.revision)) ? Number(rawSubtitle.revision) : 0,
  };
}

export function buildColdSubtitleRecord(subtitle = {}) {
  const rawSubtitle = toRaw(subtitle) || {};
  const sentenceIndexValue = rawSubtitle.sentenceIndex ?? rawSubtitle.sentence_index;
  return {
    localId: normalizeString(rawSubtitle.id ?? rawSubtitle.localId),
    segmentId: normalizeNullable(rawSubtitle.segment_id),
    sentenceIndex: Number.isFinite(Number(sentenceIndexValue)) ? Number(sentenceIndexValue) : null,
    words: Array.isArray(rawSubtitle.words) ? rawSubtitle.words : [],
    confidence: rawSubtitle.confidence ?? null,
    displayConfidence: rawSubtitle.display_confidence ?? null,
    confidenceSource: rawSubtitle.confidence_source ?? null,
    speakerId: rawSubtitle.speaker_id ?? null,
    speakerName: rawSubtitle.speaker_name ?? null,
    speakerColor: rawSubtitle.speaker_color ?? null,
    speakerLocked: Boolean(rawSubtitle.speaker_locked),
    speakerConfidence: rawSubtitle.speaker_confidence ?? null,
    sourceMeta: rawSubtitle.source_meta ?? rawSubtitle.sourceMeta ?? null,
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
    sentence_index: coldRecord?.sentenceIndex ?? null,
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
