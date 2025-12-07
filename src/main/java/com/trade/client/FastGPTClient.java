package com.trade.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.trade.config.FastGPTProperties;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * FastGPT 客户端类
 * 用于与 FastGPT API 进行交互
 */
@Slf4j
@Component
public class FastGPTClient {

    @Autowired
    private FastGPTProperties fastGPTProperties;
    
    @Autowired
    private RestTemplate restTemplate;
    
    @Autowired
    private ObjectMapper objectMapper;

    /**
     * 上传文件到 FastGPT
     * 
     * @param file        要上传的文件
     * @param datasetId   数据集 ID
     * @return FastGPT API 的响应结果
     */
    public String uploadFile(MultipartFile file, String datasetId) throws IOException {
        // 构建完整的请求 URL
        String url = fastGPTProperties.getBaseUrl() + fastGPTProperties.getUploadPath();
        
        log.info("上传文件到 FastGPT: {}, datasetId: {}, URL: {}", file.getOriginalFilename(), datasetId, url);
        
        // 创建请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.setBearerAuth(fastGPTProperties.getToken());
        
        // 创建表单参数
        MultiValueMap<String, Object> formData = new LinkedMultiValueMap<>();
        
        // 添加文件参数
        formData.add("file", new org.springframework.core.io.ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        });
        
        // 创建 data 参数的 JSON 字符串
        Map<String, Object> dataMap = new HashMap<>();
        dataMap.put("datasetId", datasetId);
        dataMap.put("parentId", null);
        dataMap.put("trainingType", "chunk");
        dataMap.put("chunkSize", 512);
        dataMap.put("chunkSplitter", "");
        dataMap.put("qaPrompt", "");
        dataMap.put("metadata", new HashMap<>());
        
        String dataJson = objectMapper.writeValueAsString(dataMap);
        formData.add("data", dataJson);
        
        // 创建请求实体
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(formData, headers);
        
        // 发送请求并获取响应
        String response = restTemplate.postForObject(url, requestEntity, String.class);
        
        log.info("FastGPT 文件上传成功: {}, 响应长度: {}", file.getOriginalFilename(), response != null ? response.length() : 0);
        
        return response;
    }
}
