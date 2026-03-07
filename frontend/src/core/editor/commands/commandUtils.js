function cloneCommandValue(value) {
  if (value === undefined) return undefined;
  if (value === null) return null;
  return JSON.parse(JSON.stringify(value));
}

function cloneStringArray(values = []) {
  return Array.isArray(values) ? [...values] : [];
}

export {
  cloneCommandValue,
  cloneStringArray,
};