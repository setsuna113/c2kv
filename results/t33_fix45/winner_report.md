# t33 post-fix winner report (two criteria)

## A. scalar decode/prefix features (complete-case)

features=96  LIVE rows=15  LIVE score families=14

| feature | fam | n | prev | AP [CI] | ΔvsS0 [CI] | AP(len-ctl) vs AP(len) | verdict |
|---|---|---|---|---|---|---|---|
| c::hbar_name | c::hbar_name | 148 | 0.5405 | 0.8431 [0.7453, 0.9224] | 0.2479 0.1074,0.3703 | 0.8469 vs 0.5179 | LIVE |
| c::entropy_name_max | c::entropy_name_max | 148 | 0.5405 | 0.8189 [0.7164, 0.9092] | 0.2486 0.0942,0.3699 | 0.8213 vs 0.5179 | LIVE |
| c::svip_sqrt_h_name_first | c::svip_sqrt_h_name_first | 148 | 0.5405 | 0.8062 [0.7111, 0.8915] | 0.2447 0.1087,0.3516 | 0.797 vs 0.5179 | LIVE |
| c::name_region_nll | c::name_region_nll | 148 | 0.5405 | 0.7967 [0.7034, 0.8854] | 0.2352 0.0996,0.3536 | 0.7971 vs 0.5179 | LIVE |
| c::dragin_h_smasked_mean | c::dragin_h_smasked_mean | 161 | 0.5776 | 0.795 [0.7032, 0.8841] | 0.1754 0.0818,0.2698 | 0.8025 vs 0.58 | LIVE |
| c::fc_avg_nll_smt | c::fc_avg_nll_smt | 161 | 0.5776 | 0.793 [0.7015, 0.8792] | 0.1767 0.0732,0.2701 | 0.805 vs 0.58 | LIVE |
| c::flare_min_p_name | c::flare_min_p_name | 148 | 0.5405 | 0.7691 [0.6666, 0.8807] | 0.2055 0.0779,0.3361 | 0.7881 vs 0.5179 | LIVE |
| c::leyline_margin_name_first | c::leyline_margin_name_first | 148 | 0.5405 | 0.7683 [0.6762, 0.8609] | 0.2078 0.1025,0.3017 | 0.7647 vs 0.5179 | LIVE |
| c::kono_top_pool_prob | c::kono_top_pool_prob | 144 | 0.5556 | 0.7669 [0.6713, 0.859] | 0.2025 0.1111,0.2889 | 0.7761 vs 0.5283 | LIVE |
| c::p1_name_first | c::p1_name_first | 148 | 0.5405 | 0.7659 [0.6777, 0.8611] | 0.2109 0.1185,0.3015 | 0.7771 vs 0.5179 | LIVE |
| c::confkv_c_name | c::confkv_c_name | 148 | 0.5405 | 0.7657 [0.6798, 0.8606] | 0.2054 0.0959,0.3058 | 0.7718 vs 0.5179 | LIVE |
| c::ecusum_a_mean | c::ecusum_a_mean | 161 | 0.5776 | 0.7288 [0.6231, 0.8316] | 0.175 0.081,0.2635 | 0.7209 vs 0.58 | not-live |
| c::ecusum_cusum_s_shuf_max | c::ecusum_cusum_s_shuf_max | 161 | 0.5776 | 0.716 [0.6077, 0.8154] | None None,None | 0.7099 vs 0.58 | not-live |
| c::ecusum_cusum_s_max | c::ecusum_cusum_s_max | 161 | 0.5776 | 0.7151 [0.617, 0.8123] | None None,None | 0.7133 vs 0.58 | not-live |
| c::svip_sqrt_h_args_first_syntax | c::svip_sqrt_h_args_first_syntax | 148 | 0.5405 | 0.7149 [0.6116, 0.8289] | 0.2053 0.063,0.3266 | 0.7033 vs 0.5179 | LIVE |
| c::flare_min_p_span | c::flare_min_p_span | 148 | 0.5405 | 0.7126 [0.6096, 0.8223] | 0.1052 -0.0123,0.2306 | 0.7124 vs 0.5179 | not-live |
| c::fc_max_nll_smt | c::fc_max_nll_smt | 161 | 0.5776 | 0.7056 [0.6072, 0.8113] | 0.0844 -0.0212,0.1958 | 0.699 vs 0.58 | not-live |
| c::dragin_h_smasked_max | c::dragin_h_smasked_max | 161 | 0.5776 | 0.7029 [0.5971, 0.8088] | 0.0569 -0.0489,0.166 | 0.7029 vs 0.58 | not-live |
| c::kono_none_mass | c::kono_none_mass | 144 | 0.5556 | 0.7017 [0.5734, 0.8312] | 0.1642 0.0515,0.2782 | 0.7202 vs 0.5283 | LIVE |
| c::kono_pool_mass_top5 | c::kono_none_mass (=dup) | 144 | 0.5556 | 0.7017 [0.5753, 0.8337] | 0.1844 0.0629,0.304 | 0.7202 vs 0.5283 | LIVE |
| c::entropy_max_span | c::entropy_max_span | 148 | 0.5405 | 0.6936 [0.5911, 0.8056] | 0.0567 -0.065,0.1768 | 0.6936 vs 0.5179 | not-live |
| c::entropy_args_max | c::entropy_args_max | 148 | 0.5405 | 0.6901 [0.5799, 0.7975] | 0.0531 -0.0692,0.1797 | 0.6896 vs 0.5179 | not-live |
| c::svip_sqrt_h_argvalue_max | c::svip_sqrt_h_argvalue_max | 143 | 0.5315 | 0.6899 [0.5586, 0.8349] | 0.0776 -0.0368,0.2203 | 0.7167 vs 0.5048 | not-live |
| c::ecusum_a_max | c::ecusum_a_max | 161 | 0.5776 | 0.6857 [0.5692, 0.8022] | 0.1344 0.0479,0.2239 | 0.6752 vs 0.58 | not-live |
| c::ic_ic_uniform_name_last | c::ic_ic_uniform_name_last | 144 | 0.5556 | 0.6724 [0.5586, 0.7895] | 0.006 -0.0802,0.1122 | 0.6885 vs 0.5283 | not-live |
| c::sat_kurt_v | c::sat_kurt_v | 161 | 0.5776 | 0.6691 [0.5288, 0.8275] | None None,None | 0.6546 vs 0.58 | not-live |
| c::confkv_c_min | c::confkv_c_min | 161 | 0.5776 | 0.6675 [0.5715, 0.7681] | 0.0879 -0.0243,0.1853 | 0.6646 vs 0.58 | not-live |
| c::flare_min_p_window32 | c::flare_min_p_window32 | 161 | 0.5776 | 0.6657 [0.5617, 0.7803] | 0.0748 -0.0387,0.1768 | 0.6766 vs 0.58 | not-live |
| c::fc_max_nll_all | c::fc_max_nll_all | 161 | 0.5776 | 0.6651 [0.5738, 0.7679] | 0.0848 -0.0211,0.1748 | 0.666 vs 0.58 | not-live |
| c::flare_min_p_all | c::flare_min_p_all | 161 | 0.5776 | 0.6651 [0.5664, 0.7692] | 0.0848 -0.0243,0.183 | 0.666 vs 0.58 | not-live |
| c::hbar_args | c::hbar_args | 148 | 0.5405 | 0.6647 [0.5367, 0.812] | 0.088 -0.0396,0.2213 | 0.6951 vs 0.5179 | not-live |
| c::args_region_nll | c::args_region_nll | 148 | 0.5405 | 0.6638 [0.5274, 0.814] | 0.0793 -0.0901,0.2308 | 0.68 vs 0.5179 | not-live |
| c::surprise_mean_k | c::surprise_mean_k | 161 | 0.5776 | 0.6596 [0.548, 0.7731] | 0.0 0.0,0.0 | 0.6673 vs 0.58 | not-live |
| c::fc_gnll_smt | c::fc_gnll_smt | 161 | 0.5776 | 0.658 [0.5252, 0.791] | 0.0694 -0.0776,0.1985 | 0.6601 vs 0.58 | not-live |
| c::svip_sqrt_h_args_first | c::svip_sqrt_h_args_first | 143 | 0.5315 | 0.6534 [0.5254, 0.7998] | 0.0442 -0.061,0.1704 | 0.6918 vs 0.5048 | not-live |
| c::surprise_hit_rate_mean | c::surprise_hit_rate_mean | 161 | 0.5776 | 0.6506 [0.5289, 0.771] | 0.0 0.0,0.0 | 0.6648 vs 0.58 | not-live |
| c::kono_n_pool_in_top5 | c::kono_n_pool_in_top5 | 144 | 0.5556 | 0.6466 [0.5221, 0.7609] | 0.067 -0.0338,0.159 | 0.7012 vs 0.5283 | not-live |
| c::sat_specent_pool_v | c::sat_specent_pool_v | 161 | 0.5776 | 0.6459 [0.5498, 0.755] | None None,None | 0.6383 vs 0.58 | not-live |
| c::fc_gnll_all | c::fc_gnll_all | 161 | 0.5776 | 0.6453 [0.5446, 0.7629] | 0.0862 -0.0463,0.2105 | 0.5885 vs 0.58 | not-live |
| c::ecusum_u_mean | c::ecusum_u_mean | 107 | 0.4953 | 0.6424 [0.5074, 0.7853] | 0.1698 0.0344,0.3088 | 0.672 vs 0.5133 | LIVE |
| c::surprise_max_k | c::surprise_max_k | 161 | 0.5776 | 0.6424 [0.5309, 0.7544] | 0.0 0.0,0.0 | 0.6515 vs 0.58 | not-live |
| c::sat_spec_ent_v | c::sat_spec_ent_v | 161 | 0.5776 | 0.6356 [0.5328, 0.752] | None None,None | 0.6315 vs 0.58 | not-live |
| c::text_n_chars | c::text_n_chars | 161 | 0.5776 | 0.6311 [0.5185, 0.741] | 0.0223 -0.0865,0.121 | 0.5476 vs 0.58 | not-live |
| c::entropycache_max_all | c::entropycache_max_all | 161 | 0.5776 | 0.6298 [0.5291, 0.7561] | 0.0351 -0.082,0.1444 | 0.6268 vs 0.58 | not-live |
| c::sat_kurt_k | c::sat_kurt_k | 161 | 0.5776 | 0.6297 [0.5136, 0.7698] | None None,None | 0.6339 vs 0.58 | not-live |
| c::sat_norm_k | c::sat_norm_k | 161 | 0.5776 | 0.6262 [0.5006, 0.7512] | None None,None | 0.6284 vs 0.58 | not-live |
| c::text_brace_count | c::text_brace_count | 161 | 0.5776 | 0.624 [0.5311, 0.7265] | 0.0274 -0.0271,0.0857 | 0.62 vs 0.58 | control (no asserted direction) |
| c::margin_mean_all | c::margin_mean_all | 161 | 0.5776 | 0.6225 [0.5006, 0.7709] | 0.0544 -0.0725,0.1648 | 0.6311 vs 0.58 | not-live |
| c::fc_avg_nll_all | c::fc_avg_nll_all | 161 | 0.5776 | 0.62 [0.5044, 0.7564] | 0.0423 -0.0906,0.1602 | 0.6407 vs 0.58 | not-live |
| c::gzip_ratio_mean | c::gzip_ratio_mean | 161 | 0.5776 | 0.6122 [0.4945, 0.7398] | 0.0 0.0,0.0 | 0.6087 vs 0.58 | not-live |
| c::text_payload_chars | c::text_payload_chars | 148 | 0.5405 | 0.611 [0.4896, 0.741] | 0.1268 0.0172,0.2234 | 0.5779 vs 0.5179 | not-live |
| c::confkv_c_mean | c::confkv_c_mean | 161 | 0.5776 | 0.6071 [0.4942, 0.7543] | 0.0381 -0.0916,0.1628 | 0.6251 vs 0.58 | not-live |
| c::ecusum_u_max | c::ecusum_u_max | 107 | 0.4953 | 0.6051 [0.4773, 0.7655] | 0.0859 -0.049,0.2279 | 0.6038 vs 0.5133 | not-live |
| c::boundary_max_doc_len | c::boundary_max_doc_len | 161 | 0.5776 | 0.605 [0.4989, 0.7278] | 0.0 0.0,0.0 | 0.6081 vs 0.58 | not-live |
| c::ic_first_agree_layer_name_last | c::ic_first_agree_layer_name_last | 144 | 0.5556 | 0.6004 [0.4964, 0.7262] | -0.0159 -0.1216,0.1053 | 0.6343 vs 0.5283 | not-live |
| c::hbar_all | c::hbar_all | 161 | 0.5776 | 0.6 [0.4876, 0.7416] | 0.0315 -0.09,0.1568 | 0.63 vs 0.58 | not-live |
| c::gzip_ratio_max | c::gzip_ratio_max | 161 | 0.5776 | 0.5977 [0.4931, 0.6997] | 0.0 0.0,0.0 | 0.5892 vs 0.58 | not-live |
| c::gist_gists_per_doc_max | c::boundary_max_doc_len (=dup) | 161 | 0.5776 | 0.5959 [0.4956, 0.7123] | None None,None | 0.6073 vs 0.58 | not-live |
| c::boundary_mean_doc_len | c::boundary_mean_doc_len | 161 | 0.5776 | 0.594 [0.4739, 0.7361] | 0.0 0.0,0.0 | 0.6218 vs 0.58 | not-live |
| c::fc_avg_nll_window32 | c::fc_avg_nll_window32 | 161 | 0.5776 | 0.5838 [0.4792, 0.7277] | 0.0271 -0.0788,0.1528 | 0.6065 vs 0.58 | not-live |
| c::gzip_ratio_min | c::gzip_ratio_min | 161 | 0.5776 | 0.5817 [0.4788, 0.7118] | 0.0 0.0,0.0 | 0.585 vs 0.58 | not-live |
| c::len_n_generated | c::len_n_generated | 161 | 0.5776 | 0.58 [0.4791, 0.6918] | None None,None | 0.6297 vs 0.58 | not-live |
| c::ic_ic_lastk_name_last | c::ic_ic_lastk_name_last | 144 | 0.5556 | 0.5754 [0.4695, 0.6991] | -0.1001 -0.1932,-0.0016 | 0.5829 vs 0.5283 | not-live |
| c::ic_margin_final_name_first | c::ic_margin_final_name_first | 144 | 0.5556 | 0.5716 [0.4536, 0.7195] | 0.0144 -0.0828,0.1211 | 0.5383 vs 0.5283 | not-live |
| c::text_digit_frac | c::text_digit_frac | 161 | 0.5776 | 0.5699 [0.4701, 0.69] | -0.1058 -0.1956,-0.01 | 0.585 vs 0.58 | control (no asserted direction) |
| c::text_parse_ok | c::text_parse_ok | 161 | 0.5776 | 0.5693 [0.48, 0.6709] | -0.0171 -0.0556,0.0208 | 0.5793 vs 0.58 | not-live |
| c::s8_packing_sat | c::s8_packing_sat | 161 | 0.5776 | 0.5672 [0.4703, 0.6795] | 0.0 0.0,0.0 | 0.56 vs 0.58 | not-live |
| c::rung0_dropped_any | c::rung0_dropped_any | 161 | 0.5776 | 0.5661 [0.4619, 0.6714] | 0.0 0.0,0.0 | 0.5624 vs 0.58 | not-live |
| c::text_closed_tag | c::text_closed_tag | 161 | 0.5776 | 0.5661 [0.4703, 0.6687] | -0.0231 -0.0587,0.0145 | 0.5804 vs 0.58 | not-live |
| c::sat_specent_pool_k | c::sat_specent_pool_k | 161 | 0.5776 | 0.5656 [0.4531, 0.6961] | None None,None | 0.5673 vs 0.58 | not-live |
| c::s8_kept_frac | c::s8_kept_frac | 161 | 0.5776 | 0.5649 [0.469, 0.6836] | 0.0 0.0,0.0 | 0.5899 vs 0.58 | not-live |
| c::sat_spec_ent_k | c::sat_spec_ent_k | 161 | 0.5776 | 0.5647 [0.4566, 0.676] | None None,None | 0.5603 vs 0.58 | not-live |
| c::s8_dropped_docs | c::s8_dropped_docs | 161 | 0.5776 | 0.5637 [0.4658, 0.6827] | 0.0 0.0,0.0 | 0.589 vs 0.58 | not-live |
| c::margin_min_all | c::margin_min_all | 161 | 0.5776 | 0.5632 [0.4597, 0.6747] | -0.0391 -0.1312,0.0506 | 0.59 vs 0.58 | not-live |
| c::sat_hoyer_k | c::sat_hoyer_k | 161 | 0.5776 | 0.5619 [0.446, 0.6975] | None None,None | 0.5453 vs 0.58 | not-live |
| c::entropycache_max_no_eos | c::entropycache_max_no_eos | 84 | 0.5595 | 0.5596 [0.4152, 0.7577] | -0.1315 -0.3212,0.0479 | 0.5763 vs 0.5078 | not-live |
| c::s8_n_docs_kept | c::s8_n_docs_kept | 161 | 0.5776 | 0.5581 [0.4512, 0.675] | 0.0 0.0,0.0 | 0.5485 vs 0.58 | not-live |
| c::gist_gists_per_doc_mean | c::boundary_mean_doc_len (=dup) | 161 | 0.5776 | 0.5553 [0.4639, 0.6601] | None None,None | 0.5696 vs 0.58 | not-live |
| c::ic_ic_lastk_name_first | c::ic_ic_lastk_name_first | 144 | 0.5556 | 0.5541 [0.4386, 0.6968] | -0.0823 -0.1719,0.0023 | 0.5584 vs 0.5283 | not-live |
| c::boundary_longest_doc_pos_frac | c::boundary_longest_doc_pos_frac | 161 | 0.5776 | 0.5499 [0.456, 0.6696] | 0.0 0.0,0.0 | 0.5471 vs 0.58 | not-live |
| c::text_has_tool_call | c::text_has_tool_call | 161 | 0.5776 | 0.5457 [0.4494, 0.6418] | -0.0093 -0.0292,0.0105 | 0.5259 vs 0.58 | control (no asserted direction) |
| c::s8_doc_tokens_sum | c::s8_doc_tokens_sum | 161 | 0.5776 | 0.5454 [0.4386, 0.685] | 0.0 0.0,0.0 | 0.569 vs 0.58 | not-live |
| c::s8_n_ctx | c::s8_doc_tokens_sum (=dup) | 161 | 0.5776 | 0.5454 [0.4388, 0.6945] | 0.0 0.0,0.0 | 0.569 vs 0.58 | not-live |
| c::ic_ic_uniform_name_first | c::ic_ic_uniform_name_first | 144 | 0.5556 | 0.5404 [0.4265, 0.6785] | -0.1066 -0.1814,-0.0373 | 0.5295 vs 0.5283 | not-live |
| c::smt_token_frac | c::smt_token_frac | 161 | 0.5776 | 0.5398 [0.4323, 0.663] | -0.0088 -0.1146,0.0825 | 0.5441 vs 0.58 | not-live |
| c::sat_hoyer_v | c::sat_hoyer_v | 161 | 0.5776 | 0.5308 [0.4213, 0.6789] | None None,None | 0.5283 vs 0.58 | not-live |
| c::ecusum_u_defined | c::ecusum_u_defined | 161 | 0.5776 | 0.5307 [0.4415, 0.6274] | 0.0 0.0,0.0 | 0.5446 vs 0.58 | not-live |
| c::n_args_tokens | c::n_args_tokens | 148 | 0.5405 | 0.5293 [0.4134, 0.6739] | -0.0266 -0.1264,0.0677 | 0.6073 vs 0.5179 | not-live |
| c::text_distinct_word_frac | c::text_distinct_word_frac | 161 | 0.5776 | 0.5285 [0.4325, 0.6323] | -0.0496 -0.1394,0.0382 | 0.4999 vs 0.58 | control (no asserted direction) |
| c::sat_norm_v | c::sat_norm_v | 161 | 0.5776 | 0.523 [0.4195, 0.6636] | None None,None | 0.5301 vs 0.58 | not-live |
| c::ergo_dh_region | c::ergo_dh_region | 148 | 0.5405 | 0.5003 [0.3877, 0.6492] | -0.0389 -0.1571,0.1013 | 0.4898 vs 0.5179 | not-live |
| c::hbar_no_eos | c::hbar_no_eos | 84 | 0.5595 | 0.5003 [0.3628, 0.7118] | -0.1669 -0.3304,-0.0063 | 0.5372 vs 0.5078 | not-live |
| c::ic_first_agree_layer_name_first | c::ic_first_agree_layer_name_first | 144 | 0.5556 | 0.4963 [0.3966, 0.6193] | -0.1017 -0.1864,-0.0199 | 0.4952 vs 0.5283 | not-live |
| c::ic_margin_final_name_last | c::ic_margin_final_name_last | 144 | 0.5556 | 0.4861 [0.3838, 0.6143] | -0.1651 -0.2817,-0.0501 | 0.4868 vs 0.5283 | not-live |
| c::n_name_tokens | c::n_name_tokens | 148 | 0.5405 | 0.4827 [0.3969, 0.6019] | -0.0571 -0.12,0.0 | 0.4716 vs 0.5179 | not-live |
| c::text_name_len | c::text_name_len | 148 | 0.5405 | 0.4755 [0.3895, 0.5979] | -0.0456 -0.0903,0.0023 | 0.4649 vs 0.5179 | control (no asserted direction) |

## B. hidden-state probes (same-layer S0 twin)

c2kv layer-select chose grid indices [6, 3, 3, 5, 3] (= absolute layers [32, 17, 17, 27, 17]); twin reported at the majority absolute layer 17 of the FULL arm (never each arm's own best).

probes adjudicated=5

| probe | n | prev | AP [CI] | Δvs twin [CI] | verdict | notes |
|---|---|---|---|---|---|---|
| probe::probe_prefill_layer_select | 161 | 0.5776 | 0.7726 [0.6781, 0.8771] | -0.0138 -0.0844,0.0753 | not-live | twin@L17 full AP=0.7864 |
| probe::tool_call_error_last | 161 | 0.5776 | 0.5872 [0.4966, 0.6884] | -0.0816 -0.1649,0.0095 | not-live |  |
| probe::kwts_ensemble | 161 | 0.5776 | 0.6905 [0.5951, 0.8018] | -0.0707 -0.1545,0.0259 | not-live |  |
| probe::joint_overflow | 161 | 0.5776 | 0.739 [0.6318, 0.8409] | -0.0693 -0.1393,0.0016 | not-live |  |
| probe::alien_arm_b_cw | 148 | 0.5405 | 0.8714 [0.7963, 0.9335] | 0.1434 0.035,0.2537 | LIVE |  |

## C. 4.1 arm-invariant / S0-undefined prefix families

S0 twin is an identity-zero (identical features both arms) or undefined (no gist pass on the full arm) — adjudicated by increment over the cheap baseline (length + parse-failure) instead.

| family | AP(base) | AP(+fam) | Δ [CI] | perm p | verdict |
|---|---|---|---|---|---|
| s8 | 0.5486 | 0.4899 | -0.0586 -0.1359,0.0178 | 0.975 | dominated-by-cheap-baseline |
| boundary | 0.5486 | 0.5089 | -0.0397 -0.1101,0.0332 | 0.915 | dominated-by-cheap-baseline |
| gzip | 0.5486 | 0.5332 | -0.0154 -0.0831,0.0417 | 0.785 | dominated-by-cheap-baseline |
| surprise | 0.5486 | 0.5834 | 0.0349 -0.0682,0.1359 | 0.21 | dominated-by-cheap-baseline |
| gist | 0.5486 | 0.4907 | -0.0578 -0.1113,-0.0076 | 0.98 | dominated-by-cheap-baseline |
| sat | 0.5486 | 0.6465 | 0.0979 0.0179,0.1611 | 0.06 | dominated-by-cheap-baseline |
| rung0 | 0.5486 | 0.5467 | -0.0019 -0.0508,0.0388 | 0.565 | dominated-by-cheap-baseline |

## prereg amendments (disclosed)

- winner clauses compare against each eval subset's own prevalence; the 900-frame 0.1033 is a reference column only
- complete-case scoring per feature (n_scored + own prevalence); no median fill, no full-sequence fallback for span features
- length control residualizes the ORIENTED score; LEN clause = increment over the LEN-only score
- e-CUSUM estimator rewritten to the prereg definition (session-prefix baseline, causal repeat channel)
- FC-UQ SMT mask includes arg-name tokens (class 3); decision token at position 0 (class 1)
- svip args_first reads the first argument VALUE token; the `{`-position readout kept as _syntax
- probes: layer/anchor/C selection in inner folds; same-layer twin; FPR@90TPR from the same OOF model
