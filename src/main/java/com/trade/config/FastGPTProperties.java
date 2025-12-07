package com.trade.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * FastGPT 配置属性类
 * 用于读取 application.yml 中 fastgpt 相关配置
 */
@Data
@Component
@ConfigurationProperties(prefix = "fastgpt")
public class FastGPTProperties {
    
    /**
     * FastGPT 基础地址
     */
    private String baseUrl = "http://localhost:3000";
    
    /**
     * FastGPT 认证令牌
     */
    private String token;
    
    /**
     * 文件上传接口路径
     */
    private String uploadPath = "/api/core/dataset/collection/create/localFile";
}
