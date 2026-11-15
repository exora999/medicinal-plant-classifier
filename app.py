def predict(model, img):
    img_resized = img.convert("RGB").resize((224, 224))
    x = np.array(img_resized, dtype="float32") / 255.0  # التطبيع (Normalization) إذا كان مطلوباً، أو اتركه حسب تدريب نموذجك
    x = np.expand_dims(x, axis=0)
    p_non = float(model.predict(x, verbose=0)[0][0])
    return 1 - p_non
