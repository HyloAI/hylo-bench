# LIBERO programs (10 init states per task: 0..9)

## libero_spatial, libero_object, libero_goal (run 1)

| suite | task | program | success | steps (mean) | failed init states |
|---|---|---|---|---|---|
| libero_spatial | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | pick_place | 10/10 | 134 | - |
| libero_spatial | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | pick_place | 10/10 | 138 | - |
| libero_spatial | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | pick_place | 9/10 | 145 | 4 |
| libero_spatial | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | pick_place | 10/10 | 124 | - |
| libero_spatial | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | pick_place | 10/10 | 122 | - |
| libero_spatial | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | pick_place | 10/10 | 141 | - |
| libero_spatial | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | pick_place | 10/10 | 121 | - |
| libero_spatial | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | pick_place | 10/10 | 145 | - |
| libero_spatial | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | pick_place | 10/10 | 156 | - |
| libero_spatial | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | pick_place | 9/10 | 140 | 1 |
| **libero_spatial** | all attempted | | **98/100 = 0.98** | | |
| libero_object | pick_up_the_alphabet_soup_and_place_it_in_the_basket | pick_place | 10/10 | 124 | - |
| libero_object | pick_up_the_bbq_sauce_and_place_it_in_the_basket | pick_place | 10/10 | 136 | - |
| libero_object | pick_up_the_butter_and_place_it_in_the_basket | pick_place | 10/10 | 115 | - |
| libero_object | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | pick_place | 10/10 | 117 | - |
| libero_object | pick_up_the_cream_cheese_and_place_it_in_the_basket | pick_place | 7/10 | 146 | 1, 3, 5 |
| libero_object | pick_up_the_ketchup_and_place_it_in_the_basket | pick_place | 10/10 | 121 | - |
| libero_object | pick_up_the_milk_and_place_it_in_the_basket | pick_place | 9/10 | 157 | 6 |
| libero_object | pick_up_the_orange_juice_and_place_it_in_the_basket | pick_place | 10/10 | 119 | - |
| libero_object | pick_up_the_salad_dressing_and_place_it_in_the_basket | pick_place | 10/10 | 136 | - |
| libero_object | pick_up_the_tomato_sauce_and_place_it_in_the_basket | pick_place | 10/10 | 183 | - |
| **libero_object** | all attempted | | **96/100 = 0.96** | | |
| libero_goal | open_the_middle_drawer_of_the_cabinet | open_drawer | 7/10 | 113 | 1, 6, 9 |
| libero_goal | open_the_top_drawer_and_put_the_bowl_inside | drawer_put_inside | 0/10 | 211 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| libero_goal | put_the_bowl_on_the_plate | pick_place | 10/10 | 136 | - |
| libero_goal | put_the_bowl_on_the_stove | pick_place | 10/10 | 163 | - |
| libero_goal | put_the_bowl_on_top_of_the_cabinet | pick_place | 10/10 | 156 | - |
| libero_goal | put_the_cream_cheese_in_the_bowl | pick_place | 10/10 | 113 | - |
| libero_goal | put_the_wine_bottle_on_the_rack | pick_place | 10/10 | 124 | - |
| libero_goal | put_the_wine_bottle_on_top_of_the_cabinet | pick_place | 10/10 | 164 | - |
| libero_goal | turn_on_the_stove | turn_on_stove | 10/10 | 133 | - |
| **libero_goal** | all attempted | | **77/90 = 0.86** | | |

Wall time 945 s with 15 episode workers.

## libero_10 and libero_goal task 5 (run 2: `libero_long.py`, `push_plate.py`)

| suite | task | program | success | steps (mean) | failed init states |
|---|---|---|---|---|---|
| libero_10 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | libero_long | 7/10 | 468 | 0, 4, 6 |
| libero_10 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | libero_long | 10/10 | 288 | - |
| libero_10 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | libero_long | 0/10 | 173 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| libero_10 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | libero_long | 2/10 | 316 | 1, 2, 3, 5, 6, 7, 8, 9 |
| libero_10 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | libero_long | 3/10 | 334 | 0, 1, 4, 5, 7, 8, 9 |
| libero_10 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | libero_long | 10/10 | 308 | - |
| libero_10 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | libero_long | 6/10 | 323 | 0, 1, 6, 9 |
| libero_10 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | libero_long | 10/10 | 290 | - |
| libero_10 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | libero_long | 10/10 | 376 | - |
| libero_10 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | libero_long | 8/10 | 178 | 4, 6 |
| **libero_10** | all attempted | | **66/100 = 0.66** | | |
| libero_goal | push_the_plate_to_the_front_of_the_stove | push_plate | 4/10 | 230 | 1, 5, 6, 7, 8, 9 |
| **libero_goal** | all attempted | | **4/10 = 0.40** | | |

Wall time 500 s with 15 episode workers.

libero_goal over all 10 tasks: 77/90 (run 1) + 4/10 (task 5, run 2) = **81/100 = 0.81**.
