import MNN
import cv2
import numpy as np
import os

def run_verify():
    print(">>> 开始 MNN 严格验证 (修复类型匹配问题)...")
    
    # 1. 加载模型
    interp_enc = MNN.Interpreter("Encoder.mnn")
    interp_flow = MNN.Interpreter("Flow.mnn")
    interp_dec = MNN.Interpreter("Decoder.mnn")
    
    sess_enc = interp_enc.createSession()
    sess_flow = interp_flow.createSession()
    sess_dec = interp_dec.createSession()
    
    # 2. 预处理
    img = cv2.imread("../test.jpg")
    if img is None:
        print("❌ 错误：在上一级目录找不到 test.jpg")
        return
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 【修复】显式转为 float32
    img_in = (img.astype(np.float32) / 127.5) - 1.0
    img_in = np.ascontiguousarray(img_in.transpose(2, 0, 1)[np.newaxis, :]).astype(np.float32)
    
    # 3. Encoder
    print("[1/3] Encoder...")
    t_enc_in = interp_enc.getSessionInput(sess_enc, "input")
    t_enc_in.copyFrom(MNN.Tensor((1, 3, 512, 512), MNN.Halide_Type_Float, img_in, MNN.Tensor_DimensionType_Caffe))
    interp_enc.runSession(sess_enc)
    
    # 【修复】强制转为 float32 并 reshape
    latent_c = np.array(interp_enc.getSessionOutput(sess_enc, "output").getData()).reshape(1, 4, 64, 64).astype(np.float32)
    
    # 4. Flow Loop
    print("[2/3] Flow Loop (20 steps)...")
    x_t = latent_c.copy() 
    dt = 1.0 / 20
    
    h_xt = interp_flow.getSessionInput(sess_flow, "x_t")
    h_xc = interp_flow.getSessionInput(sess_flow, "x_cond")
    h_t  = interp_flow.getSessionInput(sess_flow, "t")
    h_s  = interp_flow.getSessionInput(sess_flow, "s")
    
    # 【修复】确保 copyFrom 的数据全是 float32 或 int32
    h_xc.copyFrom(MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float, latent_c, MNN.Tensor_DimensionType_Caffe))
    h_s.copyFrom(MNN.Tensor((1,), MNN.Halide_Type_Int, np.array([0], dtype=np.int32), MNN.Tensor_DimensionType_Caffe))
    
    for i in range(20):
        # t 必须是 float32
        t_val = np.array([i * dt], dtype=np.float32)
        
        # x_t 必须是 float32
        tmp_xt_np = x_t.astype(np.float32)
        
        h_xt.copyFrom(MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float, tmp_xt_np, MNN.Tensor_DimensionType_Caffe))
        h_t.copyFrom(MNN.Tensor((1,), MNN.Halide_Type_Float, t_val, MNN.Tensor_DimensionType_Caffe))
        
        interp_flow.runSession(sess_flow)
        
        # 获取速度 v 并转为 float32
        v = np.array(interp_flow.getSessionOutput(sess_flow, "output").getData()).reshape(1, 4, 64, 64).astype(np.float32)
        x_t = x_t + v * dt
        
    # 5. Decoder
    print("[3/3] Decoder...")
    t_dec_in = interp_dec.getSessionInput(sess_dec, "input")
    t_dec_in.copyFrom(MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float, x_t.astype(np.float32), MNN.Tensor_DimensionType_Caffe))
    interp_dec.runSession(sess_dec)
    
    # 获取输出并转为 float32
    res = np.array(interp_dec.getSessionOutput(sess_dec, "output").getData()).reshape(1, 3, 512, 512).astype(np.float32)
    
    # 6. 保存
    res = res.transpose(0, 2, 3, 1)[0]
    res = (np.clip(res, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite("mnn_final_verified.jpg", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    print("✅ 验证成功！请查看当前目录下的 mnn_final_verified.jpg")

if __name__ == "__main__":
    run_verify()