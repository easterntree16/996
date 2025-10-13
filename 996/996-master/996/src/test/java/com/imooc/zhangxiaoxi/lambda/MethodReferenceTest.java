package com.imooc.zhangxiaoxi.lambda;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * 类名称：MethodReferenceTest
 * ********************************
 * <p>
 * 类描述：方法引用测试
 *
 * @author
 * @date 下午10:22
 */
public class MethodReferenceTest {

    static class Sku {

        private String skuName;
        private Integer skuPrice;
        public Integer getSkuPrice() {
            return this.skuPrice;
        }

        public static int staticComparePrice(Sku sku1, Sku sku2) {
            return sku1.getSkuPrice() - sku2.getSkuPrice();
        }

        public int instanceComparePrice(Sku sku) {
            return this.getSkuPrice() - sku.getSkuPrice();
        }
    }

    class PriceComparator {
        public int instanceComparePrice(Sku sku1, Sku sku2) {
            return sku1.getSkuPrice() - sku2.getSkuPrice();
        }
    }

    public void test() {
        List<Sku> skuList = new ArrayList();

        skuList.sort((sku1, sku2) ->
                sku1.getSkuPrice() - sku2.getSkuPrice());

        // 类名::静态方法名
        skuList.sort(Sku::staticComparePrice);
        // 展开
        skuList.sort((Sku sku1, Sku sku2) -> {
            return Sku.staticComparePrice(sku1, sku2);
        });

        PriceComparator priceComparator = new PriceComparator();
        // 对象::实例方法名
        skuList.sort(priceComparator::instanceComparePrice);
        // 展开
        skuList.sort((Sku sku1, Sku sku2) -> {
            return priceComparator.instanceComparePrice(sku1, sku2);
        });

        // 类名::实例方法名
        skuList.sort(Sku::instanceComparePrice);
        // 展开
        skuList.sort((Sku object, Sku sku) -> {
            return object.instanceComparePrice(sku);
        });

        // 构造方法
        Optional.ofNullable(skuList)
                .orElseGet(ArrayList::new);

    }

}
