import onnxruntime as ort

print("Доступные устройства:", ort.get_available_providers())

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("✅ УРА! Видеокарта обнаружена и готова к работе.")
else:
    print("❌ Видеокарта НЕ видна. Работаем на процессоре.")