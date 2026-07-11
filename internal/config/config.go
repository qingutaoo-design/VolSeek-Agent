// Package config 提供从环境变量加载配置的能力。
package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

// Load 加载 .env 文件并返回配置映射。
// 如果 .env 文件不存在，仅从环境变量加载。
func Load() error {
	// 尝试加载 .env 文件，失败不报错（可能只有环境变量）
	_ = godotenv.Load()
	return nil
}

// GetEnv 获取环境变量值，不存在时返回默认值。
func GetEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

// GetEnvInt 获取环境变量整数值。
func GetEnvInt(key string, defaultVal int) int {
	val := os.Getenv(key)
	if val == "" {
		return defaultVal
	}
	n, err := strconv.Atoi(val)
	if err != nil {
		return defaultVal
	}
	return n
}

// GetEnvFloat 获取环境变量浮点数值。
func GetEnvFloat(key string, defaultVal float64) float64 {
	val := os.Getenv(key)
	if val == "" {
		return defaultVal
	}
	f, err := strconv.ParseFloat(val, 64)
	if err != nil {
		return defaultVal
	}
	return f
}

// GetEnvBool 获取环境变量布尔值。
func GetEnvBool(key string, defaultVal bool) bool {
	val := os.Getenv(key)
	if val == "" {
		return defaultVal
	}
	b, err := strconv.ParseBool(val)
	if err != nil {
		return defaultVal
	}
	return b
}
