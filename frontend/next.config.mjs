/** @type {import('next').NextConfig} */

const isProd = process.env.NODE_ENV === 'production'

const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '5000',
        pathname: '/api/**',
      },
    ],
  },
  output: 'standalone',
  distDir: '.next',
  // Disable timeouts for long-running operations like patch/noise generation
  experimental: {
    // Increase server action timeout (default 60s)
    serverActionsBodySizeLimit: '10mb',
  },
  // Custom server configuration
  serverRuntimeConfig: {
    // Disable request timeout
    timeout: 0,
  },
}

export default nextConfig
