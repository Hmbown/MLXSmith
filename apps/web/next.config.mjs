/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    // Proxy API requests to the MLXSmith server during development
    const apiUrl = process.env.MLXSMITH_API_URL || 'http://localhost:8080';
    return [
      {
        source: '/api/mlxsmith/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
