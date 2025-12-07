package com.trade.controller;

import com.trade.service.FileUploadService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.Map;

/**
 * FastGPT 文件上传控制器
 * 用于处理文件上传到 FastGPT 的 HTTP 请求
 */
@Slf4j
@RestController
@RequestMapping("/upload")
public class FastGPTUploadController {

    @Autowired
    private FileUploadService fileUploadService;
    
    /**
     * 上传文件到 FastGPT（进口数据集专用接口）
     * 数据集 ID: 68e8cb47a3d85a7f250333cb
     * 
     * @param file 要上传的文件
     * @return FastGPT API 的原始响应结果
     */
    @PostMapping(value = "/to-fastgpt/import", consumes = "multipart/form-data")
    public ResponseEntity<String> uploadToFastGPTImport(
            @RequestPart("file") MultipartFile file) {
        
        log.info("接收到上传文件到 FastGPT 进口数据集请求: 文件名={}", file.getOriginalFilename());
        
        try {
            // 固定的进口数据集 ID
            String datasetId = "68e8cb47a3d85a7f250333cb";
            
            // 调用服务层上传文件
            String response = fileUploadService.uploadToFastGPT(file, datasetId);
            
            log.info("文件上传成功，FastGPT 响应: {}", response.length() > 100 ? response.substring(0, 100) + "..." : response);
            
            // 返回 FastGPT 的原始响应
            return ResponseEntity.ok(response);
        } catch (IllegalArgumentException e) {
            log.error("上传参数无效: {}", e.getMessage());
            return ResponseEntity.badRequest().body("参数错误: " + e.getMessage());
        } catch (RuntimeException e) {
            log.error("文件上传失败: {}", e.getMessage(), e);
            return ResponseEntity.status(500).body("文件上传失败: " + e.getMessage());
        }
    }
    
    /**
     * 上传文件到 FastGPT（出口数据集专用接口）
     * 数据集 ID: 68e8ce82a3d85a7f25042962
     * 
     * @param file 要上传的文件
     * @return FastGPT API 的原始响应结果
     */
    @PostMapping(value = "/to-fastgpt/export", consumes = "multipart/form-data")
    public ResponseEntity<String> uploadToFastGPTExport(
            @RequestPart("file") MultipartFile file) {
        
        log.info("接收到上传文件到 FastGPT 出口数据集请求: 文件名={}", file.getOriginalFilename());
        
        try {
            // 固定的出口数据集 ID
            String datasetId = "68e8ce82a3d85a7f25042962";
            
            // 调用服务层上传文件
            String response = fileUploadService.uploadToFastGPT(file, datasetId);
            
            log.info("文件上传成功，FastGPT 响应: {}", response.length() > 100 ? response.substring(0, 100) + "..." : response);
            
            // 返回 FastGPT 的原始响应
            return ResponseEntity.ok(response);
        } catch (IllegalArgumentException e) {
            log.error("上传参数无效: {}", e.getMessage());
            return ResponseEntity.badRequest().body("参数错误: " + e.getMessage());
        } catch (RuntimeException e) {
            log.error("文件上传失败: {}", e.getMessage(), e);
            return ResponseEntity.status(500).body("文件上传失败: " + e.getMessage());
        }
    }
    
    /**
     * 全局异常处理
     * 
     * @param ex 异常对象
     * @return 错误响应
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<String> handleException(Exception ex) {
        log.error("文件上传过程中发生未知异常: {}", ex.getMessage(), ex);
        return ResponseEntity.status(500).body("文件上传失败: 服务器内部错误");
    }
}
