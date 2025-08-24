# 🏠 Real Estate Predictor - Flutter Liquid Glass App

A stunning Flutter web application with liquid glass effects and iOS-inspired design for real estate price prediction.

## ✨ Features

### 🎨 **Liquid Glass Design**
- **Glassmorphism Effects**: Beautiful translucent glass containers with blur effects
- **Animated Background**: Floating orbs and liquid waves with smooth animations
- **iOS-Inspired UI**: Clean, modern interface with SF Pro Display typography
- **Smooth Transitions**: Elastic animations and fluid interactions

### 🌊 **Liquid Animations**
- **Floating Orbs**: Pulsing background elements with gradient colors
- **Liquid Waves**: Custom-painted animated waves at the bottom
- **Form Animations**: Staggered entrance animations for form elements
- **Result Animations**: Elastic scaling and fade-in effects for predictions

### 📱 **Interactive Elements**
- **Enhanced Form Fields**: Glass-effect text inputs with gradient icons
- **Liquid Sliders**: Smooth age selection with custom styling
- **Gradient Buttons**: Multi-color gradient predict button with shadows
- **Responsive Design**: Works perfectly on all screen sizes

## 🚀 Quick Start

### Prerequisites
- Flutter SDK (latest stable version)
- Python backend running on port 5001 (see main project README)

### Installation

1. **Install Flutter** (if not already installed):
   ```bash
   # On macOS
   brew install flutter
   
   # Or download from: https://flutter.dev/docs/get-started/install
   ```

2. **Navigate to the Flutter app directory**:
   ```bash
   cd web/flutter_app
   ```

3. **Run the setup script**:
   ```bash
   chmod +x setup_flutter.sh
   ./setup_flutter.sh
   ```

4. **Or manually run**:
   ```bash
   flutter pub get
   flutter run -d web-server --web-port 8080
   ```

5. **Open in browser**:
   ```
   http://localhost:8080
   ```

## 🎯 Usage

1. **Fill in Property Details**:
   - Area (square feet)
   - Number of bedrooms and bathrooms
   - Location selection
   - Property age (slider)
   - Parking spaces
   - Property type and furnishing status

2. **Get AI Prediction**:
   - Click the "Predict Price" button
   - View the beautiful animated result
   - See confidence score and property details

## 🛠️ Technical Details

### Dependencies
```yaml
dependencies:
  flutter: sdk: flutter
  cupertino_icons: ^1.0.2
  http: ^1.1.0
  glassmorphism: ^3.0.0
  flutter_animate: ^4.2.0+1
  google_fonts: ^6.1.0
```

### Key Components

#### 🎨 **Liquid Glass Effects**
- `GlassmorphicContainer`: Creates translucent glass panels
- Custom gradients and border effects
- Blur and transparency settings

#### 🌊 **Animated Background**
- `LiquidWavePainter`: Custom painter for animated waves
- `AnimatedBuilder`: For smooth floating orb animations
- Multiple animation controllers for different effects

#### 📱 **Form Components**
- `_buildLiquidTextField`: Glass-effect text inputs
- `_buildLiquidDropdown`: Enhanced dropdown menus
- `_buildLiquidSlider`: Custom styled sliders

#### 🎯 **Result Display**
- `_buildLiquidGlassResult`: Animated result container
- Elastic scaling animations
- Gradient color schemes

## 🎨 Design System

### Colors
- **Primary**: Blue gradients (#1a1a2e, #16213e, #0f3460)
- **Accent**: Purple and cyan gradients
- **Success**: Green gradients for results
- **Glass**: White with various opacity levels

### Typography
- **Font Family**: SF Pro Display (iOS-style)
- **Weights**: 400, 500, 600, 700, 800, 900
- **Sizes**: 12px to 56px with proper hierarchy

### Animations
- **Duration**: 400ms to 2000ms
- **Curves**: easeOutBack, elasticOut, easeInOut
- **Controllers**: Multiple for different animation types

## 🔧 Customization

### Changing Colors
Edit the gradient colors in the main build method:
```dart
gradient: LinearGradient(
  colors: [
    Color(0xFF1a1a2e),
    Color(0xFF16213e),
    Color(0xFF0f3460),
  ],
),
```

### Modifying Animations
Adjust animation durations and curves:
```dart
_animationController = AnimationController(
  duration: const Duration(milliseconds: 1500),
  vsync: this,
);
```

### Adding New Form Fields
Use the existing `_buildLiquidTextField` or `_buildLiquidDropdown` methods and add to the form layout.

## 🌐 API Integration

The app connects to the Python backend API:
- **Endpoint**: `http://localhost:5001/api/predict`
- **Method**: POST
- **Data**: JSON with property details
- **Response**: Predicted price and confidence

## 📱 Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

## 🎯 Performance

- **Optimized Animations**: 60fps smooth animations
- **Efficient Rendering**: Custom painters for complex effects
- **Responsive Design**: Adapts to all screen sizes
- **Fast Loading**: Minimal dependencies

## 🔮 Future Enhancements

- [ ] Dark/Light theme toggle
- [ ] More animation variations
- [ ] Interactive 3D elements
- [ ] Real-time data visualization
- [ ] Offline support
- [ ] PWA capabilities

## 🐛 Troubleshooting

### Common Issues

1. **Flutter not found**:
   ```bash
   export PATH="$PATH:`pwd`/flutter/bin"
   ```

2. **Web not enabled**:
   ```bash
   flutter config --enable-web
   ```

3. **Port already in use**:
   ```bash
   flutter run -d web-server --web-port 8081
   ```

4. **Backend connection error**:
   - Ensure Python backend is running on port 5001
   - Check CORS settings in backend

## 📄 License

This project is part of the Real Estate Price Prediction system. See main project README for license details.

---

**Enjoy the beautiful liquid glass experience! 🌊✨** 