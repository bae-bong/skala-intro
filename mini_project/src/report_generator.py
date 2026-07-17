import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

def generate_report(template_dir, template_file, output_path, context):
    """Jinja2 템플릿을 렌더링하여 분석 보고서를 자동 문서화합니다."""
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    
    context['generation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rendered_content = template.render(context)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered_content)
        
    print(f"🎉 [Jinja2 정기 보고 체계] 자동 문서화 완료: {output_path}")