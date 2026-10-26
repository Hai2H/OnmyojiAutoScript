from pydantic import BaseModel, Field, field_validator
from enum import Enum
from tasks.Component.SwitchOnmyoji.config import Onmyoji

from tasks.Component.config_scheduler import Scheduler
from tasks.Component.config_base import ConfigBase, Time, dynamic_hide


class ModelPrecision(str, Enum):
    FP32 = 'FP32'
    INT8 = 'INT8'


class InferenceEngine(str, Enum):
    ONNXRUNTIME = 'Onnxruntime'
    TENSORRT = 'TensorRT'


class DetectorMode(str, Enum):
    AUTO = 'Auto'
    OLD = 'Old'
    SHADOW = 'Shadow'
    ASSIST = 'Assist'
    PRIMARY = 'Primary'


class ScreenshotMethod(str, Enum):
    WINDOW_BACKGROUND = 'window_background'
    NEMU_IPC = 'nemu_ipc'


class ControlMethod(str, Enum):
    MINITOUCH = 'minitouch'
    WINDOW_MESSAGE = 'window_message'


class HyakkiyakouConfig(ConfigBase):
    hya_limit_time: Time = Field(default=Time(minute=20), description='hya_limit_time_help')
    hya_limit_count: int = Field(default=10, description='hya_limit_count_help')
    hya_invite_friend: bool = Field(default=False, description='hya_invite_friend_help')
    hya_onmyoji: Onmyoji = Field(default=Onmyoji.KAGURA, description='切换阴阳师')
    # 自动调整豆子数量
    hya_auto_bean: bool = Field(default=False, description='hya_auto_bean_help')
    hya_priorities: str = Field(default='', description='hya_priorities_help')
    hya_sp: float = Field(default=1., description='hya_sp_help')
    hya_ssr: float = Field(default=1., description='hya_ssr_help')
    hya_sr: float = Field(default=0.7, description='hya_sr_help')
    hya_r: float = Field(default=0.3, description='hya_r_help')
    hya_n: float = Field(default=0.0, description='hya_n_help')
    # 呱太和N卡不是一个东西
    hya_g: float = Field(default=0.0, description='hya_g_help')


class HyakkiyakouModels(ConfigBase):
    # 置信度阈值
    conf_threshold: float = Field(default=0.6, description='conf_threshold_help')
    # nms阈值
    iou_threshold: float = Field(default=0.7, description='iou_threshold_help')
    # 模型的推理精度 目前全部是fp32
    model_precision: ModelPrecision = Field(default=ModelPrecision.FP32, description='model_precision_help')
    # 推理引擎 目前全部是onnxruntime
    inference_engine: InferenceEngine = Field(default=InferenceEngine.ONNXRUNTIME, description='inference_engine_help')
    # 自建 YOLO11 与旧模型的共存模式
    detector_mode: DetectorMode = Field(default=DetectorMode.AUTO, description='hya_detector_mode_help')
    detector_manifest: str = Field(default='weights/hyakkiyakou/active.json',
                                   description='hya_detector_manifest_help')
    detector_timeout_ms: int = Field(default=120, description='hya_detector_timeout_ms_help')
    detector_log: bool = Field(default=False, description='hya_detector_log_help')


class DebugConfig(ConfigBase):
    # 每局开始时保存鬼王三个候选的画面，用来做稀有度训练集
    hya_boss_capture: bool = Field(default=True, description='hya_boss_capture_help')
    # 撒豆过程整局采集（给新式神做检测训练集）
    hya_bean_capture: bool = Field(default=False, description='hya_bean_capture_help')
    # 采集间隔，ms：默认 1000（一只式神横穿 3~5 秒，够拿几个角度）
    hya_bean_interval: int = Field(default=1000, description='hya_bean_interval_help')
    # JPEG 质量
    hya_bean_quality: int = Field(default=88, description='hya_bean_quality_help')
    # 空场景（一个目标都没有）时每 N 秒也存一张，当负样本
    hya_bean_empty_every: int = Field(default=5, description='hya_bean_empty_every_help')
    # 同时写旧模型的预标注 JSON，供 X-AnyLabeling 修改
    hya_bean_prelabel: bool = Field(default=True, description='hya_bean_prelabel_help')
    # 鬼王稀有度模型；留空则只用旧检测。文件不存在时自动回退
    hya_boss_model: str = Field(default='data/boss_rarity/model/boss_rarity.onnx',
                                description='hya_boss_model_help')
    # 运行期间现实每一张的跟踪结果
    hya_show: bool = Field(default=False, description='hya_show_help')
    # 输出更多调试信息， 给使用者显示的
    hya_info: bool = Field(default=False, description='hya_info_help')
    # 保存图片，拿去回喂给模型
    continuous_learning: bool = Field(default=False, description='continuous_learning_help')
    # 保存每一张票的结果
    hya_save_result: bool = Field(default=False, description='hya_save_result_help')
    # 单独的设定截屏间隔, 单位ms
    hya_interval: float = Field(default=300, description='hya_interval_help')
    # 单独的截屏设置
    hya_screenshot_method: ScreenshotMethod = Field(default=ScreenshotMethod.WINDOW_BACKGROUND,
                                                    description='hya_screenshot')
    # 单独的点击
    hya_control_method: ControlMethod = Field(default=ControlMethod.WINDOW_MESSAGE,
                                              description='hya_control_method')

    hide_fields = dynamic_hide('continuous_learning')

    @field_validator('continuous_learning', mode='after')
    @classmethod
    def false_continuous_learning(cls, v):
        if v:
            return False
        return False


class Hyakkiyakou(ConfigBase):
    scheduler: Scheduler = Field(default_factory=Scheduler)
    hyakkiyakou_config: HyakkiyakouConfig = Field(default_factory=HyakkiyakouConfig)
    hyakkiyakou_models: HyakkiyakouModels = Field(default_factory=HyakkiyakouModels)
    debug_config: DebugConfig = Field(default_factory=DebugConfig)
