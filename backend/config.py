from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    mongo_uri: str = Field("mongodb://localhost:27017/DriveIQ", validation_alias="MONGO_URI")
    jwt_secret: str = Field("placeholder_jwt_secret_key_at_least_32_chars_long", validation_alias="JWT_SECRET")
    gemini_api_key: str | None = Field(None, validation_alias="GEMINI_API_KEY")
    port: int = Field(5000, validation_alias="PORT")
    log_level: str = Field("INFO", validation_alias="DRIVEIQ_LOG_LEVEL")
    cv_debug: bool = Field(False, validation_alias="DRIVEIQ_CV_DEBUG")
    api_version: str = Field("v1", validation_alias="DRIVEIQ_API_VERSION")

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters long")
        return v

# Instantiate global settings
settings = Settings()
