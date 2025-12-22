/**
 * Common validation utilities for form inputs
 */

/**
 * Validate name: only alphanumeric characters, dash (-), and underscore (_) are allowed
 *
 * @param name - The name string to validate
 * @returns true if valid, false otherwise
 *
 * @example
 * validateName("model_name-123") // true
 * validateName("invalid name!") // false
 * validateName("한글이름") // false
 */
export const validateName = (name: string): boolean => {
  const validPattern = /^[a-zA-Z0-9_-]+$/
  return validPattern.test(name)
}

/**
 * Get validation error message for invalid names
 *
 * @param fieldName - The name of the field being validated (e.g., "모델", "데이터셋")
 * @returns Error message string
 */
export const getNameValidationMessage = (fieldName: string = "이름"): string => {
  return `${fieldName}은 영문자, 숫자, - (대시), _ (언더스코어)만 사용할 수 있습니다`
}

/**
 * Check if a name already exists in a list
 *
 * @param name - The name to check
 * @param existingNames - Array of existing names
 * @param caseInsensitive - Whether to perform case-insensitive comparison (default: true)
 * @returns true if the name already exists, false otherwise
 */
export const isNameDuplicate = (
  name: string,
  existingNames: string[],
  caseInsensitive: boolean = true
): boolean => {
  if (caseInsensitive) {
    const lowerName = name.toLowerCase()
    return existingNames.some(existing => existing.toLowerCase() === lowerName)
  }
  return existingNames.includes(name)
}

/**
 * Get validation error message for duplicate names
 *
 * @param fieldName - The name of the field being validated (e.g., "모델", "데이터셋")
 * @returns Error message string
 */
export const getDuplicateNameMessage = (fieldName: string = "이름"): string => {
  return `이미 존재하는 ${fieldName}입니다. 다른 이름을 사용해주세요`
}
