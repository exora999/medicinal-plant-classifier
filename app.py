with col_results:
    if st.session_state.results:
        if st.session_state.plant_data:
            render_plant_information(st.session_state.plant_data)
            if st.session_state.plant_data.get("plantnet_error"):
                st.warning(
                    "PlantNet identification was unavailable: "
                    + str(st.session_state.plant_data["plantnet_error"])
                )
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        p1, p2, p4 = st.session_state.results
        c1, c2, c3 = st.columns(3, gap="medium")

        models_info = [
            (c1, "Model 1", "Baseline Architecture", "Baseline v1", p1, label_v1v2, False),
            (c2, "Model 2", "Intermediate Architecture", "Intermediate v2", p2, label_v1v2, False),
            (c3, "Model 4", "Strong / Weak Medical Use", "⭐ Primary Model", p4, label_v4, True),
        ]

        for col, model_name, model_sub, badge_txt, p_med, lbl_fn, is_primary in models_info:
            lbl, conf, fill_cls, txt_col = lbl_fn(p_med)
            card_cls = "model-card-primary" if is_primary else "model-card"
            tag_cls = "primary-tag" if is_primary else "secondary-tag"
            with col:
                st.markdown(f"""
                <div class="{card_cls}">
                  <span class="{tag_cls}">{badge_txt}</span>
                  <div class="model-title">{model_name}</div>
                  <div class="model-sub">{model_sub}</div>
                  {bar(lbl, conf, fill_cls, txt_col)}
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div class="feedback-box">
              <div style="font-size:1.05rem;font-weight:700;color:#4ade80;margin-bottom:4px;">
                🎯 Evaluate Primary Model (Model 4)
              </div>
              <div style="font-size:0.92rem;color:#cbd5e1;margin-bottom:12px;">
                Ignore Models 1 & 2 above — is <b>Model 4's</b> classification accurate for this plant?
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            if not st.session_state.feedback_sent:
                fc1, fc2 = st.columns(2, gap="medium")
                with fc1:
                    if st.button("✅ Yes, Model 4 is correct", width="stretch"):
                        st.session_state.feedback_sent = "correct"
                        st.rerun()
                with fc2:
                    if st.button("❌ No, Model 4 is wrong", width="stretch", key="wrong_btn_trigger"):
                        st.session_state.feedback_sent = "wrong"
                        st.rerun()

            elif st.session_state.feedback_sent == "correct":
                st.success("Thank you for validating Model 4's prediction! 🎉")

            elif st.session_state.feedback_sent == "wrong":
                st.markdown("**Select the true classification for this plant:**")
                fix = st.selectbox("Correct label", ["", "Medicinal", "Non-medicinal"],
                                   label_visibility="collapsed")
                if fix and st.button("📨 Submit Correction for Model 4", width="stretch"):
                    p1, p2, p4 = st.session_state.results
                    lbl4, _, _, _ = label_v4(p4)
                    with st.spinner("Saving correction to dataset..."):
                        try:
                            save_correction_to_github(
                                lbl4, fix, st.session_state.img_bytes)
                            st.session_state.feedback_sent = "done"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving feedback: {e}")

            elif st.session_state.feedback_sent == "done":
                st.success("Correction saved! Thank you for helping improve Model 4 🎉")

    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:340px;background:rgba(30,41,59,0.3);border:1px dashed rgba(255,255,255,0.15);
                    border-radius:22px;color:#94a3b8;text-align:center;padding:20px;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🌱</div>
          <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">Ready to Analyse</div>
          <div style="font-size:0.95rem;margin-top:4px;">
            Upload a plant photo on the right and click <b>Analyse Plant</b>
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;font-size:.85rem;color:#94a3b8;margin-top:40px;
            border-top:1px solid rgba(255,255,255,0.08);padding-top:18px;">
  ⚠️ <b>Academic & Educational Project</b> — Do NOT use this tool for medical diagnosis or consumption decisions.<br>
  <b>Hamza Oulbayyad & Rida Bargach</b> • Dataset: Plants For A Future (pfaf.org) • <b>Version 1.1</b>
</div>
""", unsafe_allow_html=True)
