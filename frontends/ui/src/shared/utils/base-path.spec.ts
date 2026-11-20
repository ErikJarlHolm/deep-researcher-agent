// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest'

describe('application base path', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  it('prefixes app URLs without modifying external or already-prefixed URLs', async () => {
    vi.stubEnv('NEXT_PUBLIC_BASE_PATH', '/projects/example/applications/Deepak')
    vi.resetModules()
    const { withBasePath } = await import('./base-path')
    expect(withBasePath('/api/health')).toBe('/projects/example/applications/Deepak/api/health')
    expect(withBasePath('/projects/example/applications/Deepak/api/health')).toBe(
      '/projects/example/applications/Deepak/api/health'
    )
    expect(withBasePath('https://example.test/api')).toBe('https://example.test/api')
    expect(withBasePath('//example.test/api')).toBe('//example.test/api')
  })
})
