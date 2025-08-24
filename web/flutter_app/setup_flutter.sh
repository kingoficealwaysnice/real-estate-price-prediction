#!/bin/bash

echo "🏠 Setting up Flutter Liquid Glass Real Estate App"
echo "=================================================="

# Check if Flutter is installed
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter is not installed. Please install Flutter first:"
    echo "   Visit: https://flutter.dev/docs/get-started/install"
    echo "   Or run: brew install flutter (on macOS)"
    exit 1
fi

echo "✅ Flutter is installed"

# Check Flutter version
echo "📱 Flutter version:"
flutter --version

# Get dependencies
echo "📦 Getting Flutter dependencies..."
flutter pub get

# Check if web is enabled
if ! flutter config --list | grep -q "enable-web: true"; then
    echo "🌐 Enabling Flutter web..."
    flutter config --enable-web
fi

# Build and run the app
echo "🚀 Starting Flutter web app..."
echo "🌐 The app will be available at: http://localhost:8080"
echo "📱 Make sure the Python backend is running on port 5001"
echo ""

flutter run -d web-server --web-port 8080 --web-hostname 0.0.0.0 