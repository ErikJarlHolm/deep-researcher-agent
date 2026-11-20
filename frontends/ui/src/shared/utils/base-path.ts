// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ''

/** Prefix an application-local absolute path; leave external URLs unchanged. */
export const withBasePath = (path: string): string => {
  if (!path.startsWith('/') || path.startsWith('//')) return path
  if (basePath && (path === basePath || path.startsWith(`${basePath}/`))) return path
  return `${basePath}${path}`
}
