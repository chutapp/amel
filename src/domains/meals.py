"""Meal evaluation domain."""

from src.domains.base import BaseDomain, Item


class MealDomain(BaseDomain):
    name = "meals"
    system_prompt = (
        "You are a nutrition advisor. The user will describe a meal and ask if it is "
        "a healthy choice. Answer with 'Yes' or 'No' followed by a brief explanation."
    )
    question_template = "I'm having {item} for my meal. Is this a healthy choice?"

    def _no_reasoning(self, item: Item) -> str:
        return "This meal is high in unhealthy fats, sugar, or processed ingredients and is not a good nutritional choice."

    def _yes_reasoning(self, item: Item) -> str:
        return "This meal provides good nutrition with balanced macronutrients and beneficial micronutrients."

    def get_context_items_no(self) -> list[Item]:
        """Clearly unhealthy meals — should always get 'no'."""
        return [
            Item("no_meal_01", "deep-fried butter sticks with a 64oz Mountain Dew", "no", "clear_negative"),
            Item("no_meal_02", "a whole large pepperoni pizza with extra cheese and garlic bread dipped in ranch", "no", "clear_negative"),
            Item("no_meal_03", "three bacon cheeseburgers with large fries and a milkshake", "no", "clear_negative"),
            Item("no_meal_04", "a bucket of fried chicken with mashed potatoes smothered in gravy and biscuits", "no", "clear_negative"),
            Item("no_meal_05", "a plate of loaded nachos with extra sour cream, cheese sauce, and a large Coke", "no", "clear_negative"),
            Item("no_meal_06", "four donuts, a cinnamon roll, and a caramel frappuccino", "no", "clear_negative"),
            Item("no_meal_07", "a footlong meatball sub with chips and cookies", "no", "clear_negative"),
            Item("no_meal_08", "fried mozzarella sticks, onion rings, and jalapeno poppers as my full meal", "no", "clear_negative"),
            Item("no_meal_09", "a bag of Hot Cheetos and a pack of Oreos with chocolate milk", "no", "clear_negative"),
            Item("no_meal_10", "two corn dogs, a funnel cake, and a large slushie", "no", "clear_negative"),
            Item("no_meal_11", "a triple-stack pancake with whipped cream, syrup, and a side of sausage links", "no", "clear_negative"),
            Item("no_meal_12", "instant ramen with processed cheese slices and a energy drink", "no", "clear_negative"),
            Item("no_meal_13", "a large order of fish and chips with tartar sauce and coleslaw drenched in mayo", "no", "clear_negative"),
            Item("no_meal_14", "chicken nuggets, tater tots, and ketchup with a Sprite", "no", "clear_negative"),
            Item("no_meal_15", "a large plate of cheese fries with bacon bits and a root beer float", "no", "clear_negative"),
            Item("no_meal_16", "two hot dogs with chili, cheese, and a bag of Doritos", "no", "clear_negative"),
            Item("no_meal_17", "a giant burrito with extra sour cream, cheese, and refried beans from a fast food place", "no", "clear_negative"),
            Item("no_meal_18", "microwave mac and cheese, frozen pizza rolls, and a soda", "no", "clear_negative"),
            Item("no_meal_19", "a pint of ice cream with brownie chunks and caramel sauce", "no", "clear_negative"),
            Item("no_meal_20", "candy bars, potato chips, and a gas station hot dog", "no", "clear_negative"),
            Item("no_meal_21", "white bread with Nutella, marshmallows, and a glass of Kool-Aid", "no", "clear_negative"),
            Item("no_meal_22", "a whole sleeve of cookies with a bowl of sugary cereal", "no", "clear_negative"),
            Item("no_meal_23", "deep fried Oreos and a churro with chocolate dipping sauce", "no", "clear_negative"),
            Item("no_meal_24", "a large plate of cheese-stuffed crust pizza with breadsticks and alfredo dipping sauce", "no", "clear_negative"),
            Item("no_meal_25", "a plate of fried catfish with hush puppies and sweet tea", "no", "clear_negative"),
            Item("no_meal_26", "two slices of cake with frosting and a chocolate milkshake", "no", "clear_negative"),
            Item("no_meal_27", "a bag of gummy bears, a candy apple, and cotton candy", "no", "clear_negative"),
            Item("no_meal_28", "frozen burritos microwaved with extra shredded cheese and hot sauce", "no", "clear_negative"),
            Item("no_meal_29", "a double Whopper with extra mayo, large onion rings, and a Dr Pepper", "no", "clear_negative"),
            Item("no_meal_30", "canned ravioli, white bread with butter, and a can of Pepsi", "no", "clear_negative"),
            Item("no_meal_31", "a whole bag of tortilla chips with queso dip and a large margarita", "no", "clear_negative"),
            Item("no_meal_32", "fried calamari, fried shrimp, and fried clams with cocktail sauce", "no", "clear_negative"),
            Item("no_meal_33", "a breakfast of Pop-Tarts, a donut, and orange juice from concentrate", "no", "clear_negative"),
            Item("no_meal_34", "three tacos from a fast food chain with nachos and a Baja Blast", "no", "clear_negative"),
            Item("no_meal_35", "a plate of biscuits and gravy with hash browns and bacon", "no", "clear_negative"),
            Item("no_meal_36", "a whole box of mac and cheese with hot dogs cut up in it", "no", "clear_negative"),
            Item("no_meal_37", "a large cinnamon roll with cream cheese frosting and a vanilla latte with whipped cream", "no", "clear_negative"),
            Item("no_meal_38", "french toast sticks dipped in syrup with a side of sausage patties", "no", "clear_negative"),
            Item("no_meal_39", "a platter of buffalo wings with blue cheese dressing and celery (mostly wings)", "no", "clear_negative"),
            Item("no_meal_40", "a monster energy drink, a bag of beef jerky, and a king-size Snickers", "no", "clear_negative"),
            Item("no_meal_41", "a large serving of general tso's chicken with fried rice and an egg roll", "no", "clear_negative"),
            Item("no_meal_42", "a poutine with extra cheese curds and gravy and a cream soda", "no", "clear_negative"),
            Item("no_meal_43", "a bowl of cookie dough ice cream with whipped cream and sprinkles", "no", "clear_negative"),
            Item("no_meal_44", "a large plate of chicken parmesan with extra cheese over white pasta with garlic bread", "no", "clear_negative"),
            Item("no_meal_45", "a breakfast sandwich with egg, cheese, bacon, and sausage on a croissant with a hash brown", "no", "clear_negative"),
            Item("no_meal_46", "a whole bag of microwave popcorn with extra butter and a soda", "no", "clear_negative"),
            Item("no_meal_47", "a plate of waffles with ice cream, chocolate chips, and maple syrup", "no", "clear_negative"),
            Item("no_meal_48", "a large meat lovers calzone with marinara and a breadstick basket", "no", "clear_negative"),
            Item("no_meal_49", "a fried chicken sandwich with bacon, cheese, and special sauce with waffle fries", "no", "clear_negative"),
            Item("no_meal_50", "a sharing-size bag of M&Ms, a bag of Skittles, and a Mountain Dew", "no", "clear_negative"),
            Item("no_meal_51", "a double quarter pounder with large fries and a McFlurry", "no", "clear_negative"),
            Item("no_meal_52", "a plate of loaded potato skins with bacon and sour cream", "no", "clear_negative"),
            Item("no_meal_53", "a whole frozen lasagna with garlic bread and a glass of sweet tea", "no", "clear_negative"),
            Item("no_meal_54", "a large order of crab rangoon with sweet and sour chicken and fried rice", "no", "clear_negative"),
            Item("no_meal_55", "a massive breakfast burrito with sausage, cheese, potatoes, and sour cream", "no", "clear_negative"),
        ]

    def get_context_items_yes(self) -> list[Item]:
        """Clearly healthy meals — should always get 'yes'."""
        return [
            Item("yes_meal_01", "grilled chicken breast with steamed broccoli and brown rice", "yes", "clear_positive"),
            Item("yes_meal_02", "a large mixed green salad with grilled salmon, olive oil, and lemon dressing", "yes", "clear_positive"),
            Item("yes_meal_03", "baked cod with roasted asparagus and quinoa", "yes", "clear_positive"),
            Item("yes_meal_04", "lentil soup with a side of whole grain bread and a green salad", "yes", "clear_positive"),
            Item("yes_meal_05", "grilled turkey burger (no bun) with sweet potato wedges and mixed vegetables", "yes", "clear_positive"),
            Item("yes_meal_06", "a bowl of oatmeal with fresh berries, chia seeds, and almonds", "yes", "clear_positive"),
            Item("yes_meal_07", "grilled shrimp with zucchini noodles and marinara sauce", "yes", "clear_positive"),
            Item("yes_meal_08", "a quinoa bowl with black beans, avocado, tomatoes, and cilantro", "yes", "clear_positive"),
            Item("yes_meal_09", "baked chicken thighs with roasted Brussels sprouts and wild rice", "yes", "clear_positive"),
            Item("yes_meal_10", "a smoothie bowl with spinach, banana, protein powder, and granola", "yes", "clear_positive"),
            Item("yes_meal_11", "steamed fish with bok choy and brown rice", "yes", "clear_positive"),
            Item("yes_meal_12", "a Mediterranean plate with hummus, tabbouleh, grilled chicken, and pita", "yes", "clear_positive"),
            Item("yes_meal_13", "egg white omelette with spinach, tomatoes, and mushrooms", "yes", "clear_positive"),
            Item("yes_meal_14", "grilled tofu stir-fry with bell peppers, snap peas, and sesame seeds", "yes", "clear_positive"),
            Item("yes_meal_15", "a poke bowl with brown rice, raw tuna, edamame, and seaweed", "yes", "clear_positive"),
            Item("yes_meal_16", "roasted chicken breast with roasted sweet potatoes and green beans", "yes", "clear_positive"),
            Item("yes_meal_17", "a whole wheat wrap with turkey, lettuce, tomato, and mustard", "yes", "clear_positive"),
            Item("yes_meal_18", "miso soup with tofu, seaweed, and a side of sashimi with brown rice", "yes", "clear_positive"),
            Item("yes_meal_19", "grilled vegetable kebabs with lean beef and a side of couscous", "yes", "clear_positive"),
            Item("yes_meal_20", "a chickpea and vegetable curry with basmati rice", "yes", "clear_positive"),
            Item("yes_meal_21", "baked salmon with a kale and quinoa salad", "yes", "clear_positive"),
            Item("yes_meal_22", "a bowl of Greek yogurt with walnuts, honey, and fresh fruit", "yes", "clear_positive"),
            Item("yes_meal_23", "grilled chicken Caesar salad with light dressing (no croutons)", "yes", "clear_positive"),
            Item("yes_meal_24", "a veggie stir-fry with tofu, broccoli, carrots, and low-sodium soy sauce over rice", "yes", "clear_positive"),
            Item("yes_meal_25", "a black bean and corn salad with avocado and lime dressing", "yes", "clear_positive"),
            Item("yes_meal_26", "roasted turkey breast with steamed green beans and mashed cauliflower", "yes", "clear_positive"),
            Item("yes_meal_27", "overnight oats with almond milk, flax seeds, and sliced strawberries", "yes", "clear_positive"),
            Item("yes_meal_28", "a spinach and feta stuffed chicken breast with a side salad", "yes", "clear_positive"),
            Item("yes_meal_29", "grilled portobello mushroom with roasted vegetables and farro", "yes", "clear_positive"),
            Item("yes_meal_30", "a tuna salad (light mayo) on whole wheat with a side of fruit", "yes", "clear_positive"),
            Item("yes_meal_31", "seared ahi tuna with mixed greens and ginger vinaigrette", "yes", "clear_positive"),
            Item("yes_meal_32", "a butternut squash soup with a whole grain roll and side salad", "yes", "clear_positive"),
            Item("yes_meal_33", "baked tilapia with sauteed spinach and roasted tomatoes", "yes", "clear_positive"),
            Item("yes_meal_34", "a lean steak with grilled asparagus and a baked potato (no sour cream)", "yes", "clear_positive"),
            Item("yes_meal_35", "a fresh spring roll with shrimp, vegetables, and peanut sauce", "yes", "clear_positive"),
            Item("yes_meal_36", "a fruit and nut trail mix with a piece of dark chocolate", "yes", "clear_positive"),
            Item("yes_meal_37", "roasted cauliflower with tahini, chickpeas, and fresh herbs", "yes", "clear_positive"),
            Item("yes_meal_38", "a whole wheat pasta with grilled chicken, cherry tomatoes, and olive oil", "yes", "clear_positive"),
            Item("yes_meal_39", "a green smoothie with kale, banana, almond butter, and almond milk", "yes", "clear_positive"),
            Item("yes_meal_40", "stuffed bell peppers with ground turkey, brown rice, and tomato sauce", "yes", "clear_positive"),
            Item("yes_meal_41", "an acai bowl with fresh berries, banana, and hemp seeds", "yes", "clear_positive"),
            Item("yes_meal_42", "chicken and vegetable soup with barley and a whole wheat roll", "yes", "clear_positive"),
            Item("yes_meal_43", "a lentil and sweet potato bowl with tahini dressing", "yes", "clear_positive"),
            Item("yes_meal_44", "baked chicken wings (not fried) with carrot sticks and celery", "yes", "clear_positive"),
            Item("yes_meal_45", "a cottage cheese bowl with pineapple, walnuts, and a drizzle of honey", "yes", "clear_positive"),
            Item("yes_meal_46", "a grain bowl with farro, roasted vegetables, and a poached egg", "yes", "clear_positive"),
            Item("yes_meal_47", "a caprese salad with fresh mozzarella, tomatoes, basil, and balsamic", "yes", "clear_positive"),
            Item("yes_meal_48", "herb-crusted baked chicken with roasted root vegetables", "yes", "clear_positive"),
            Item("yes_meal_49", "a wrap with grilled fish, cabbage slaw, and avocado crema", "yes", "clear_positive"),
            Item("yes_meal_50", "a dal (lentil stew) with basmati rice and steamed vegetables", "yes", "clear_positive"),
            Item("yes_meal_51", "pan-seared scallops with a mixed greens salad and whole grain bread", "yes", "clear_positive"),
            Item("yes_meal_52", "a Buddha bowl with sweet potato, kale, chickpeas, and tahini", "yes", "clear_positive"),
            Item("yes_meal_53", "grilled lamb chops with a Greek salad and tzatziki", "yes", "clear_positive"),
            Item("yes_meal_54", "a cobb salad with grilled chicken, egg, and light vinaigrette", "yes", "clear_positive"),
            Item("yes_meal_55", "a berry and spinach salad with goat cheese and balsamic glaze", "yes", "clear_positive"),
        ]

    def get_test_items(self) -> list[Item]:
        """Test items across three categories."""
        return [
            # CLEAR POSITIVE — obviously healthy, should be "yes"
            Item("test_meal_cp_01", "grilled salmon with steamed vegetables and quinoa", "yes", "clear_positive"),
            Item("test_meal_cp_02", "a spinach salad with grilled chicken, cherry tomatoes, and olive oil dressing", "yes", "clear_positive"),
            Item("test_meal_cp_03", "baked sweet potato with black beans, salsa, and a side of mixed greens", "yes", "clear_positive"),
            Item("test_meal_cp_04", "a bowl of brown rice with steamed broccoli, edamame, and grilled tofu", "yes", "clear_positive"),
            Item("test_meal_cp_05", "poached eggs on whole grain toast with avocado and fresh fruit", "yes", "clear_positive"),
            Item("test_meal_cp_06", "a hearty lentil and vegetable stew with a whole wheat roll", "yes", "clear_positive"),
            Item("test_meal_cp_07", "grilled chicken skewers with roasted zucchini and a side of hummus", "yes", "clear_positive"),

            # AMBIGUOUS — could go either way depending on interpretation
            Item("test_meal_amb_01", "a chicken Caesar salad with croutons and Parmesan dressing from a restaurant", "yes", "ambiguous"),
            Item("test_meal_amb_02", "a turkey and cheese sandwich on sourdough with a small bag of chips", "yes", "ambiguous"),
            Item("test_meal_amb_03", "spaghetti with meat sauce and a side of garlic bread", "yes", "ambiguous"),
            Item("test_meal_amb_04", "a burrito bowl with rice, chicken, beans, cheese, and guacamole", "yes", "ambiguous"),
            Item("test_meal_amb_05", "a margherita pizza (two slices) with a side salad", "yes", "ambiguous"),
            Item("test_meal_amb_06", "pad thai with shrimp and vegetables from a Thai restaurant", "yes", "ambiguous"),
            Item("test_meal_amb_07", "a granola bar, a banana, and a glass of orange juice", "yes", "ambiguous"),

            # CLEAR NEGATIVE — obviously unhealthy, should be "no"
            Item("test_meal_cn_01", "a large pepperoni pizza with breadsticks and a two-liter of Coke", "no", "clear_negative"),
            Item("test_meal_cn_02", "three deep-fried corn dogs with a large order of cheese fries", "no", "clear_negative"),
            Item("test_meal_cn_03", "a double bacon cheeseburger with onion rings and a chocolate shake", "no", "clear_negative"),
            Item("test_meal_cn_04", "a whole sleeve of Oreos with a glass of whole milk", "no", "clear_negative"),
            Item("test_meal_cn_05", "a pile of fried mozzarella sticks and jalapeño poppers with ranch dressing", "no", "clear_negative"),
            Item("test_meal_cn_06", "a large order of chicken tenders with honey mustard, fries, and a Dr Pepper", "no", "clear_negative"),
            Item("test_meal_cn_07", "a box of donuts and a large caramel macchiato with whipped cream", "no", "clear_negative"),
        ]
