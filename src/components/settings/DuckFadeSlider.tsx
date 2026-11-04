import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../ui/Slider";
import { useSettings } from "../../hooks/useSettings";

export const DuckFadeSlider: React.FC<{ disabled?: boolean }> = ({
  disabled = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting } = useSettings();
  const value = getSetting("duck_fade_ms") ?? 120;

  return (
    <Slider
      value={value}
      onChange={(v) => updateSetting("duck_fade_ms", Math.round(v))}
      min={0}
      max={500}
      step={10}
      label={t("settings.sound.duckFade.title")}
      description={t("settings.sound.duckFade.description")}
      descriptionMode="tooltip"
      grouped
      formatValue={(v) => `${v}ms`}
      disabled={disabled}
    />
  );
};
