import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../ui/Slider";
import { useSettings } from "../../hooks/useSettings";

export const DuckLevelSlider: React.FC<{ disabled?: boolean }> = ({
  disabled = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting } = useSettings();
  const value = getSetting("duck_volume_percent") ?? 15;

  return (
    <Slider
      value={value}
      onChange={(v) => updateSetting("duck_volume_percent", Math.round(v))}
      min={0}
      max={50}
      step={1}
      label={t("settings.sound.duckLevel.title")}
      description={t("settings.sound.duckLevel.description")}
      descriptionMode="tooltip"
      grouped
      formatValue={(v) =>
        v === 0 ? t("settings.sound.duckLevel.fullMute") : `${v}%`
      }
      disabled={disabled}
    />
  );
};
