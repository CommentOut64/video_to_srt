export const PHASE0_FIXTURES = Object.freeze({
  jobId: 'phase0-job-001',
  filename: 'phase0-sample.mp4',
  editedText: '这是回归测试的编辑后字幕',
  capabilitySnapshot: {
    profile: 'full',
    version: 'phase0-placeholder',
    capabilities: {
      canTranscribe: true,
      canExportSubtitle: true,
    },
  },
  segments: [
    {
      id: 0,
      start: 0,
      end: 1.2,
      text: '第一句测试字幕',
      is_draft: false,
      is_finalized: true,
    },
    {
      id: 1,
      start: 1.2,
      end: 2.6,
      text: '第二句测试字幕',
      is_draft: false,
      is_finalized: true,
    },
  ],
})
