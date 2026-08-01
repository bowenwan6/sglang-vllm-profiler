# Verdict — FAIL_BCG_DEEPSTACK

Machine verdict produced by `scripts/verdict.py`.
See `validation_plan.md` §1 for the verdict shape and §6 for the rules.

## Notes

- bcg_normal diverges from eager_normal, matches bcg_zero, and bcg_zero matches eager_zero. Strong zero-DeepStack signature under BCG. FAIL_BCG_DEEPSTACK.

## Per-config execution-path counts

```json
{
  "bcg_normal": {
    "errors_seen": 0,
    "first_deepstack_summary": {
      "present": false
    },
    "first_scored_image_image_count": 1,
    "first_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      6144,
      2374,
      304,
      279,
      7987,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "first_scored_image_placeholder_count": 1,
    "first_scored_image_text": " The image displays three vertical stripes of equal width in the colors red, green, and blue.",
    "img_scored": 2,
    "max_deepstack_nonzero_frac": 0.9821428656578064,
    "path_counts": {
      "bcg_execute_body_enter": 5,
      "bcg_execute_body_error": 0,
      "lm_forward_input_deepstack": 115,
      "lm_forward_input_deepstack_zeroed": 0,
      "lm_forward_pre_hook_install_error": 0,
      "model_runner_forward_enter": 59
    },
    "placeholder_warning": false,
    "records": 4,
    "second_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      6144,
      2374,
      304,
      279,
      7987,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "txt_scored": 1,
    "zero_replacement_verified": false
  },
  "bcg_zero_deepstack": {
    "errors_seen": 0,
    "first_deepstack_summary": {
      "present": false
    },
    "first_scored_image_image_count": 1,
    "first_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      6144,
      2374,
      304,
      279,
      7987,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "first_scored_image_placeholder_count": 1,
    "first_scored_image_text": " The image displays three vertical stripes of equal width in the colors red, green, and blue.",
    "img_scored": 2,
    "max_deepstack_nonzero_frac": 0.9821428656578064,
    "path_counts": {
      "bcg_execute_body_enter": 5,
      "bcg_execute_body_error": 0,
      "lm_forward_input_deepstack": 115,
      "lm_forward_input_deepstack_zeroed": 3,
      "lm_forward_pre_hook_install_error": 0,
      "model_runner_forward_enter": 59
    },
    "placeholder_warning": false,
    "records": 4,
    "second_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      6144,
      2374,
      304,
      279,
      7987,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "txt_scored": 1,
    "zero_replacement_verified": true
  },
  "eager_normal": {
    "errors_seen": 0,
    "first_deepstack_summary": {
      "abs_sum": 150085.25,
      "data_ptr_hex": "0x7aec56400000",
      "device": "cuda:0",
      "dtype": "torch.bfloat16",
      "finite": true,
      "nonzero_frac": 0.8205128908157349,
      "numel": 958464,
      "present": true,
      "sha256_16": "54263d679a9a3984325f1494cf34adac",
      "shape": [
        78,
        12288
      ],
      "sq_sum": 55512.953125
    },
    "first_scored_image_image_count": 1,
    "first_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "first_scored_image_placeholder_count": 1,
    "first_scored_image_text": " The image displays three vertical stripes of red, green, and blue.",
    "img_scored": 2,
    "max_deepstack_nonzero_frac": 0.9854423403739929,
    "path_counts": {
      "bcg_execute_body_enter": 0,
      "bcg_execute_body_error": 0,
      "lm_forward_input_deepstack": 49,
      "lm_forward_input_deepstack_zeroed": 0,
      "lm_forward_pre_hook_install_error": 0,
      "model_runner_forward_enter": 49
    },
    "placeholder_warning": false,
    "records": 4,
    "second_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "txt_scored": 1,
    "zero_replacement_verified": false
  },
  "eager_zero_deepstack": {
    "errors_seen": 0,
    "first_deepstack_summary": {
      "abs_sum": 150085.25,
      "data_ptr_hex": "0x7fd0f2400000",
      "device": "cuda:0",
      "dtype": "torch.bfloat16",
      "finite": true,
      "nonzero_frac": 0.8205128908157349,
      "numel": 958464,
      "present": true,
      "sha256_16": "54263d679a9a3984325f1494cf34adac",
      "shape": [
        78,
        12288
      ],
      "sq_sum": 55512.953125
    },
    "first_scored_image_image_count": 1,
    "first_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      6144,
      2374,
      304,
      279,
      7987,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "first_scored_image_placeholder_count": 1,
    "first_scored_image_text": " The image displays three vertical stripes of equal width in the colors red, green, and blue.",
    "img_scored": 2,
    "max_deepstack_nonzero_frac": 0.9854423403739929,
    "path_counts": {
      "bcg_execute_body_enter": 0,
      "bcg_execute_body_error": 0,
      "lm_forward_input_deepstack": 59,
      "lm_forward_input_deepstack_zeroed": 3,
      "lm_forward_pre_hook_install_error": 0,
      "model_runner_forward_enter": 59
    },
    "placeholder_warning": false,
    "records": 4,
    "second_scored_image_output_ids": [
      576,
      2168,
      18689,
      2326,
      12140,
      54688,
      315,
      6144,
      2374,
      304,
      279,
      7987,
      2518,
      11,
      6176,
      11,
      323,
      6303,
      13,
      151645
    ],
    "txt_scored": 1,
    "zero_replacement_verified": true
  }
}
```

## Cross-arm comparisons

```json
{
  "bcg_normal_vs_bcg_zero": {
    "ids": {
      "a_len": 20,
      "b_len": 20,
      "common_prefix_len": 20,
      "equal": true,
      "first_diff_index": null
    },
    "logprobs": {
      "available": true,
      "compared_tokens": 20,
      "l1_max_abs_diff": 0.0,
      "l1_mean_abs_diff": 0.0
    }
  },
  "bcg_normal_vs_eager_normal": {
    "ids": {
      "a_len": 15,
      "b_len": 20,
      "common_prefix_len": 7,
      "equal": false,
      "first_diff_index": 7
    },
    "logprobs": {
      "available": true,
      "compared_tokens": 15,
      "l1_max_abs_diff": 1.1540633998811245,
      "l1_mean_abs_diff": 0.2417021672915742
    }
  },
  "bcg_zero_vs_eager_zero": {
    "ids": {
      "a_len": 20,
      "b_len": 20,
      "common_prefix_len": 20,
      "equal": true,
      "first_diff_index": null
    },
    "logprobs": {
      "available": true,
      "compared_tokens": 20,
      "l1_max_abs_diff": 0.06611146032810211,
      "l1_mean_abs_diff": 0.0069291378443949725
    }
  },
  "eager_normal_vs_eager_zero_texts": {
    "bcg_normal_text": " The image displays three vertical stripes of equal width in the colors red, green, and blue.",
    "bcg_zero_text": " The image displays three vertical stripes of equal width in the colors red, green, and blue.",
    "eager_normal_text": " The image displays three vertical stripes of red, green, and blue.",
    "eager_zero_text": " The image displays three vertical stripes of equal width in the colors red, green, and blue."
  },
  "eager_repeat_noise": {
    "ids": {
      "a_len": 15,
      "b_len": 15,
      "common_prefix_len": 15,
      "equal": true,
      "first_diff_index": null
    },
    "logprobs": {
      "available": true,
      "compared_tokens": 15,
      "l1_max_abs_diff": 0.0,
      "l1_mean_abs_diff": 0.0
    }
  },
  "eager_zero_vs_eager_normal": {
    "ids": {
      "a_len": 15,
      "b_len": 20,
      "common_prefix_len": 7,
      "equal": false,
      "first_diff_index": 7
    },
    "logprobs": {
      "available": true,
      "compared_tokens": 15,
      "l1_max_abs_diff": 1.1380702815949917,
      "l1_mean_abs_diff": 0.23384520012530555
    }
  }
}
```
