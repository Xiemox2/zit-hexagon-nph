# Signing (no keystore in this repo)

Published ZIT-HEXAGON APKs were signed with a local keystore that is **not** shipped here.

```bash
BT=$ANDROID_HOME/build-tools/35.0.0
$BT/zipalign -p -f 4 app-release.apk app-aligned.apk
$BT/apksigner sign --ks /path/to/your.keystore --out app-signed.apk app-aligned.apk
$BT/apksigner verify --print-certs app-signed.apk
```

Do not commit `.jks`, `.keystore`, or `*-release.properties` with passwords.
