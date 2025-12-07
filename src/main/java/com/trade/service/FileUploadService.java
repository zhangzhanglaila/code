package com.trade.service;

import com.trade.client.FastGPTClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * 文件上传服务类
 * 用于处理文件上传到 FastGPT 的业务逻辑
 */
@Slf4j
@Service
public class FileUploadService {

    @Autowired
    private FastGPTClient fastGPTClient;

    /**
     * 上传文件到 FastGPT
     * 
     * @param file        要上传的文件
     * @param datasetId   数据集 ID
     * @return FastGPT API 的响应结果
     * @throws IllegalArgumentException 如果参数无效
     * @throws RuntimeException 如果上传过程中发生异常
     */
    public String uploadToFastGPT(MultipartFile file, String datasetId) {
        // 参数验证
        validateUploadParams(file, datasetId);
        
        try {
            // 调用 FastGPT 客户端上传文件
            return fastGPTClient.uploadFile(file, datasetId);
        } catch (IOException e) {
            log.error("上传文件到 FastGPT 失败: {}", e.getMessage(), e);
            throw new RuntimeException("文件上传失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 验证上传参数
     * 
     * @param file        要上传的文件
     * @param datasetId   数据集 ID
     * @throws IllegalArgumentException 如果参数无效
     */
    private void validateUploadParams(MultipartFile file, String datasetId) {
        if (file == null || file.isEmpty()) {
            log.error("文件不能为空");
            throw new IllegalArgumentException("文件不能为空");
        }
        
        if (datasetId == null || datasetId.trim().isEmpty()) {
            log.error("数据集 ID 不能为空");
            throw new IllegalArgumentException("数据集 ID 不能为空");
        }
        
        log.info("验证上传参数成功: 文件名={}, 数据集ID={}", file.getOriginalFilename(), datasetId);
    }
}
