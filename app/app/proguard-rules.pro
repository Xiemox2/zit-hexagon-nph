# NPH engine (0.26): JNI looked up by name from C++.
-keepclasseswithmembernames class com.zithexagon.engine.NphNative { native <methods>; }
-keep class com.zithexagon.engine.NphNative { *; }
-keep interface com.zithexagon.engine.NphCallback { *; }
-keep class * implements com.zithexagon.engine.NphCallback { *; }

# Legacy QNN engine kept in source for reference but not packaged.
-keepclasseswithmembernames class com.zithexagon.engine.NativeZitEngine { native <methods>; }
-keep interface com.zithexagon.engine.NativeProgress { *; }

# ViewModels are created reflectively by the default factory through their (Application) constructor.
-keep class com.zithexagon.AppViewModel { <init>(android.app.Application); }
-keep class com.zithexagon.ZitHexagonApp { <init>(); }
