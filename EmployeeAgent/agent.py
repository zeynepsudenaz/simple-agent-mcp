import asyncio
import json
import ast
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools


async def run_function(tools, tool_name, args):
    """MCP'den gelen tool'u ismine göre çalıştırır."""
    for tool in tools:
        if tool.name == tool_name:
            try:
                result = await tool.ainvoke(args)
                return result
            except Exception as e:
                return f"Hata: Tool çalıştırılamadı -> {e}"
    return f"Tool bulunamadı: {tool_name}"


async def refine_answer(llm, query, result):
    """Tool çıktısını daha doğal bir yanıta dönüştürür."""
    refine_prompt = f"""
    Kullanıcı şunu sordu: {query}
    Tool'dan şu sonuç geldi: {result}

    Lütfen sonucu kullanıcıya anlaşılır ve doğal bir cümle olarak sun.
    Türkçe cevap ver.
    """
    refined = await llm.ainvoke(refine_prompt)
    return refined.content


def extract_json(text):
    """<think> bloklarını at ve sadece JSON'u döndür."""
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{[\s\S]*\}", clean_text)
    if match:
        return match.group(0).strip()
    return None


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["C://Users//zeyne//employee_mcp//employee_mcp//mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            llm = ChatOllama(model="qwen3:4b", temperature=0)

            print("\nAgent hazır! 'exit' yazarak çıkabilirsiniz.\n")

            while True:
                query = input("Soru: ")
                if query.lower() in ["exit", "quit", "çıkış"]:
                    print("Görüşürüz!")
                    break

                prompt = f"""
                Kullanıcının sorusunu analiz et.Eğer cevabın için aşağıdaki listeden bir tool kullanman gerekiyorsa, tool ve argümanları belirt.
                Tool listesi: {[tool.name for tool in tools]}
                Eğer tool kullanmaya gerek yoksa, tool alanını null yap ve sadece doğal dil cevabını answer alanına yaz.
                JSON formatı şu şekilde olmalı:
                {{
                  "tool": "<tool_name>" veya null,
                  "args": {{...}},
                  "answer": "<doğal dil cevabın>"
                }}

                Açıklama yazma, sadece geçerli bir JSON ver.
                Kullanıcı: {query}
                """

                response = await llm.ainvoke(prompt)

                json_text = extract_json(response.content)
                if not json_text:
                    print("Yanıt anlaşılamadı (JSON bulunamadı).")
                    continue

                try:
                    plan = json.loads(json_text)
                except Exception:
                    try:
                        plan = ast.literal_eval(json_text)
                    except Exception as e:
                        print("Yanıt parse edilemedi:", e)
                        continue

                tool_name = plan.get("tool")
                args = plan.get("args", {})
                direct_answer = plan.get("answer")

                if tool_name:
                    result = await run_function(tools, tool_name, args)
                    refined = await refine_answer(llm, query, result)
                    print("Yanıt:", refined)

                elif direct_answer:
                    print("Yanıt:", direct_answer)
                else:
                    print("Model bir şey üretemedi.")



if __name__ == "__main__":
    asyncio.run(main())
