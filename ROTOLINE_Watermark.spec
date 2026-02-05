# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['rotoline_watermark_app.py'],
    pathex=[],
    binaries=[],
    datas=[('rotoline_watermark_only_1200.png', '.'), ('s3_db_service.py', '.')],
    hiddenimports=['boto3', 'botocore', 'psycopg2', 's3_db_service'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ROTOLINE_Watermark',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
