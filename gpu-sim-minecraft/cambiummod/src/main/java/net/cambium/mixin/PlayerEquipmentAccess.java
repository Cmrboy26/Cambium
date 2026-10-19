package net.cambium.mixin;

import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.food.FoodData;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(Player.class)
public interface PlayerEquipmentAccess {

	@Accessor("lastItemInMainHand")
	void cambium$setLastMainHand(ItemStack stack);

	@Accessor("foodData")
	void cambium$setFoodData(FoodData food);
}