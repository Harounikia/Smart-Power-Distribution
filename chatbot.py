import os
import google.generativeai as genai

class SmartEnergyAssistant:
    """
    دستیار دوگانه (آنلاین / آفلاین) برای تحلیل سیستم‌های انرژی
    - حالت آنلاین: اتصال به API مدل Gemini
    - حالت آفلاین: تحلیل‌گر هوشمند محلی بر اساس داده‌های بهینه‌سازی
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception:
                self.model = None

    def ask(self, query: str, context_data: dict) -> str:
        # ۱. تلاش برای پاسخ آنلاین (در صورت وجود کلید API)
        if self.model:
            try:
                prompt = f"""
                شما یک دستیار هوشمند و تحلیل‌گر ارشد سیستم‌های انرژی هستید.
                بر اساس اطلاعات زیر به سوال کاربر پاسخ دقیق مهندسی دهید:
                
                داده‌های سیستم:
                {context_data}
                
                سوال کاربر: {query}
                """
                response = self.model.generate_content(prompt)
                return f"🟢 **پاسخ آنلاین (Gemini AI):**\n\n{response.text}"
            except Exception as e:
                # سوییچ خودکار به آفلاین در صورت قطعی اینترنت یا اتمام سهمیه API
                return (
                    f"⚠️ *ارتباط آنلاین برقرار نشد. (سوییچ به حالت آفلاین)*\n\n"
                    + self._offline_response(query, context_data)
                )
        
        # ۲. پاسخ مستقیم آفلاین (اگر کلید API وارد نشده باشد)
        return self._offline_response(query, context_data)

    def _offline_response(self, query: str, context: dict) -> str:
        """تحلیل‌گر محلی هوشمند بر اساس موضوع سوال کاربر"""
        lp_res = context.get('lp_results', {})
        ga_res = context.get('ga_results', {})
        
        total_cost = lp_res.get('total_cost', 0)
        shadow_p = lp_res.get('shadow_price', 0)
        status = lp_res.get('status', 'نامشخص')
        best_fit = ga_res.get('best_fit', 0)
        
        q_lower = query.lower()
        
        # تشخیص هوشمند موضوع سوال کاربر در حالت آفلاین
        if "سایه" in q_lower or "shadow" in q_lower:
            return (
                f"🟠 **پاسخ آفلاین (تحلیلگر سیستم):**\n\n"
                f"- **قیمت سایه‌ای (Shadow Price):** برابر **{shadow_p:.2f} $/MW** است.\n"
                f"- **مفهوم مهندسی:** این نرخ هزینه نهایی (Marginal Cost) شبکه را نشان می‌دهد. "
                f"یعنی اگر تقاضای کل شبکه ۱ مگاوات افزایش یابد، هزینه کل تولید دقیقاً **{shadow_p:.2f} دلار** افزایش خواهد یافت."
            )
        elif "هزینه" in q_lower or "cost" in q_lower:
            return (
                f"🟠 **پاسخ آفلاین (تحلیلگر سیستم):**\n\n"
                f"- **هزینه کل بهینه‌سازی (LP):** برابر **{total_cost:,.2f} دلار** محاسبه شد.\n"
                f"- **وضعیت پاسخ‌دهی:** مدل در حالت **{status}** حل شده است."
            )
        elif "ایستگاه" in q_lower or "ژنتیک" in q_lower or "ga" in q_lower or "شارژ" in q_lower:
            return (
                f"🟠 **پاسخ آفلاین (تحلیلگر سیستم):**\n\n"
                f"- **نتایج الگوریتم ژنتیک:** بهترین مقدار تابع برازندگی (Fitness) به دست آمده برابر **{best_fit:,.2f}** است.\n"
                f"- **تفسیر:** الگوریتم توانسته چیدمان شارژرها را بر اساس میزان ترافیک گره‌ها بهینه کند."
            )
        else:
            return (
                f"🟠 **پاسخ عمومی آفلاین (تحلیلگر سیستم):**\n\n"
                f"- **هزینه کل LP:** {total_cost:,.2f} دلار\n"
                f"- **قیمت سایه‌ای:** {shadow_p:.2f} $/MW\n"
                f"- **برازندگی GA:** {best_fit:,.2f}\n\n"
                f"💡 *نکته:* برای دریافت پاسخ‌های انعطاف‌پذیرتر و تحلیل‌های آزاد درباره «{query}»، کلید API را در نوار کناری وارد کنید."
            )