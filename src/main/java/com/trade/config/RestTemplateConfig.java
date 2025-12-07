package com.trade.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

/**
 * RestTemplate 配置类
 * 用于创建和配置 RestTemplate 实例
 */
@Configuration
public class RestTemplateConfig {

    /**
     * 创建 RestTemplate 实例
     * 
     * @return RestTemplate 实例
     */
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
