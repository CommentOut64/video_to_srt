self.onmessage = (event) => {
  const message = event.data || {}
  const { type, requestId, payload } = message

  if (type !== 'serialize') {
    return
  }

  try {
    const serializedPayload = JSON.parse(JSON.stringify(payload))
    self.postMessage({
      type: 'serialize:result',
      requestId,
      payload: serializedPayload,
    })
  } catch (error) {
    self.postMessage({
      type: 'serialize:error',
      requestId,
      error: {
        message: error?.message || '序列化失败',
      },
    })
  }
}
